#!/usr/bin/env python3
"""Freeze a trained predictor. Train a residual ranker on top of mode_head.

s_k = V11_logits_k + r_theta(z_k). Only r_theta is trained.
z_k = traj summary (end, 4s, yaw, lateral) + agent/lane/inter tokens.
Label: ADE-winner index. GT is not an input.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.hyp_ranker import HypRanker
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.cached_collate import CachedWindowCollate, try_cached_loader
from src.train_motion_prediction_v10 import WindowSampleCollateV10
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v10 import _list_or_fallback, model_forward


def ade_per_mode(pred, future, future_valid):
    disp = torch.linalg.vector_norm(pred - future[:, None, :, :], dim=-1)
    mask = future_valid[:, None, :].float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (disp * mask).sum(dim=-1) / denom


def ranker_forward(ranker, pred, samples, prior_logits):
    return ranker(
        pred,
        samples.get("agent_tok"),
        samples.get("type_idx"),
        samples.get("lane_tok"),
        samples.get("inter_tok"),
        prior_logits,
    )


def ranking_loss(scores, winner, margin=0.5, margin_weight=1.0):
    ce = F.cross_entropy(scores, winner)
    s_star = scores.gather(1, winner.view(-1, 1))
    hinge = (margin - (s_star - scores)).clamp_min(0)
    mask = torch.ones_like(hinge)
    mask.scatter_(1, winner.view(-1, 1), 0.0)
    hinge_mean = (hinge * mask).sum(dim=1).mean()
    return ce + margin_weight * hinge_mean, ce, hinge_mean


@torch.no_grad()
def predict_hyps(model, samples, amp_dtype, device):
    # Frozen predictor in fp32. Ranker head may still use AMP.
    with torch.amp.autocast("cuda", enabled=False):
        pred, goals, logits = model_forward(model, samples)
    return pred.float(), goals.float(), logits.float()


def ranker_ok(metrics, ade6_limit=0.70, pick_acc_min=0.7, gap_max=0.15):
    a6 = metrics.get("val_minade6", 99.0)
    a1 = metrics.get("val_minade1", 99.0)
    acc = metrics.get("val_pick_acc", 0.0)
    return a6 < ade6_limit and acc > pick_acc_min and (a1 - a6) < gap_max


@torch.no_grad()
def evaluate_ranker(model, ranker, val_loader, device, amp_dtype, max_steps=None):
    model.eval()
    ranker.eval()
    # Must be three tensors. `a = b = c = torch.zeros()` aliases one object,
    # so += on each name summed ADE6+ADE1_base+ADE1_rank into every metric (~3.35).
    ade6 = torch.zeros((), device=device)
    ade1_base = torch.zeros((), device=device)
    ade1_rk = torch.zeros((), device=device)
    n = 0
    agree = 0
    for step, samples in enumerate(val_loader, start=1):
        if max_steps and step > max_steps:
            break
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        pred, _, base_logits = predict_hyps(model, samples, amp_dtype, device)
        ade = ade_per_mode(pred, samples["future"], samples["future_valid"])
        winner = ade.argmin(dim=-1)
        rk_logits = ranker_forward(ranker, pred, samples, base_logits)
        b = pred.shape[0]
        ix = torch.arange(b, device=device)
        ade6 += ade[ix, winner].sum()
        ade1_base += ade[ix, base_logits.argmax(-1)].sum()
        ade1_rk += ade[ix, rk_logits.argmax(-1)].sum()
        agree += int((rk_logits.argmax(-1) == winner).sum())
        n += b
    if n == 0:
        return {}
    a6 = float(ade6 / n)
    a1 = float(ade1_rk / n)
    return {
        "val_minade6": a6,
        "val_minade1_base": float(ade1_base / n),
        "val_minade1": a1,
        "val_error_score": 0.5 * (a1 + a6),
        "val_pick_acc": agree / n,
        "n": n,
    }


def load_predictor(arch: str, ckpt_path: str, device):
    if arch == "v12":
        from src.train_motion_prediction_v12 import MotionPredictorV12
        model = MotionPredictorV12(hidden=256, modes=6)
    else:
        from src.train_motion_prediction_v11 import MotionPredictorV11
        model = MotionPredictorV11(hidden=256, modes=6)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v11", "v12"], default="v11")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v11_ranker_residual")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-scenes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--max-packs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.amp != "fp32" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 80)
    print(" RANKER residual — s = V11 mode_head logits + r_theta(z)")
    print(f" arch={args.arch}  ckpt={args.ckpt}")
    print(" z: traj summary (end/4s/yaw/lat) + agent/lane/inter. Flatten off.")
    print(f" loss: CE + {args.margin_weight:g} * hinge(m={args.margin:g}). GT is not an input.")
    print("=" * 80, flush=True)

    model = load_predictor(args.arch, args.ckpt, device)
    ranker = HypRanker().to(device)
    opt = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ds = try_cached_loader(args.cache_root, "train", args.max_packs)
    val_ds = try_cached_loader(args.cache_root, "val", args.max_packs)
    pin = device.type == "cuda"
    if train_ds is not None and val_ds is not None:
        print(f"-> using collate cache {args.cache_root}", flush=True)
        train_collate = CachedWindowCollate(args.max_targets, True)
        val_collate = CachedWindowCollate(args.max_targets, False)
    else:
        print("-> cache missing, live collate (slow CPU)", flush=True)
        train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
        val_paths = _list_or_fallback(args.data_root, "val", args.max_packs, fallback=train_paths)
        probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
        n_windows = max(1, len(probe.get("windows", [])))
        train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
        val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
        train_collate = WindowSampleCollateV10(args.max_targets, True)
        val_collate = WindowSampleCollateV10(args.max_targets, False)
    train_loader = make_loader(
        train_ds, args.batch_scenes, args.workers, args.prefetch, True, train_collate, pin,
    )
    val_loader = make_loader(
        val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, pin,
    )
    n_train = len(train_ds.paths) if hasattr(train_ds, "paths") else len(train_ds)
    n_val = len(val_ds.paths) if hasattr(val_ds, "paths") else len(val_ds)
    print(f"-> Train {n_train}  Val {n_val}  ranker params {sum(p.numel() for p in ranker.parameters())}", flush=True)

    history = []
    best_ade1 = float("inf")
    log_file = os.path.join(args.out_dir, "training.log")
    metrics_file = os.path.join(args.out_dir, "metrics.json")
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    def log_val(tag, rec):
        msg = (
            f"{tag} | "
            f"Val ADE6 {rec.get('val_minade6', 0):.4f} ADE1_base {rec.get('val_minade1_base', 0):.4f} "
            f"ADE1_rank {rec.get('val_minade1', 0):.4f} acc {rec.get('val_pick_acc', 0):.3f} "
            f"Error {rec.get('val_error_score', 0):.4f}"
        )
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return msg

    val0 = evaluate_ranker(model, ranker, val_loader, device, amp_dtype, args.max_val_steps or None)
    rec0 = {"epoch": 0, "train_loss": None, "train_pick_acc": None, "epoch_sec": 0.0, **val0}
    history.append(rec0)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    log_val("Ranker 00/pre-train (residual=0, expect ADE1_rank≈ADE1_base≈1.25)", rec0)
    if val0.get("val_minade6", 99.0) >= 0.70:
        print(
            f"ABORT: frozen decoder val ADE6 {val0.get('val_minade6')} "
            f"(need < 0.70, expect ~0.63). Check forward, not the ranker head.",
            flush=True,
        )
        with open(os.path.join(args.out_dir, "success.json"), "w", encoding="utf-8") as f:
            json.dump({"success": False, "reason": "pretrain_ade6_too_high", **val0}, f, indent=2)
        return
    prior_gap = abs(val0.get("val_minade1", 99.0) - val0.get("val_minade1_base", 0.0))
    if prior_gap > 0.20:
        print(
            f"ABORT: ADE1_rank {val0.get('val_minade1')} vs ADE1_base {val0.get('val_minade1_base')} "
            f"(gap {prior_gap:.3f} > 0.20). Prior logits not wired; residual is not ~0.",
            flush=True,
        )
        with open(os.path.join(args.out_dir, "success.json"), "w", encoding="utf-8") as f:
            json.dump({"success": False, "reason": "pretrain_prior_not_used", **val0}, f, indent=2)
        return

    for epoch in range(1, args.epochs + 1):
        ranker.train()
        model.eval()
        t0 = time.time()
        loss_sum = 0.0
        acc_sum = 0.0
        n_bat = 0
        for step, samples in enumerate(train_loader, start=1):
            if args.max_train_steps and step > args.max_train_steps:
                break
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            pred, _, base_logits = predict_hyps(model, samples, amp_dtype, device)
            ade = ade_per_mode(pred, samples["future"], samples["future_valid"])
            winner = ade.argmin(dim=-1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                scores = ranker_forward(ranker, pred.detach(), samples, base_logits.detach())
                loss, _, _ = ranking_loss(scores, winner, args.margin, args.margin_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach())
            acc_sum += float((scores.argmax(-1) == winner).float().mean().detach())
            n_bat += 1
            if step % 200 == 0:
                print(
                    f" [Ranker {epoch:02d}/{args.epochs}] step {step:05d} loss {loss_sum/max(n_bat,1):.4f} "
                    f"pick_acc {acc_sum/max(n_bat,1):.3f}",
                    flush=True,
                )
        val = evaluate_ranker(model, ranker, val_loader, device, amp_dtype, args.max_val_steps or None)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_bat, 1),
            "train_pick_acc": acc_sum / max(n_bat, 1),
            "epoch_sec": time.time() - t0,
            **val,
        }
        history.append(rec)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        log_val(
            f"Ranker {epoch:02d}/{args.epochs} ({rec['epoch_sec']:.1f}s) | "
            f"pick_acc {rec['train_pick_acc']:.3f}",
            rec,
        )
        payload = {"epoch": epoch, "ranker_state": ranker.state_dict(), "metrics": rec, "predictor_ckpt": args.ckpt, "arch": args.arch}
        torch.save(payload, os.path.join(args.out_dir, "last.pth"))
        if val.get("val_minade1", 1e9) < best_ade1:
            best_ade1 = val["val_minade1"]
            torch.save(payload, os.path.join(args.out_dir, "best_ranker.pth"))
            print(f"[NEW BEST ranker ADE1] {best_ade1:.4f} (ADE6 {val['val_minade6']:.4f})", flush=True)
        if args.max_train_steps:
            break

    trained = [h for h in history if h.get("epoch", 0) >= 1]
    if trained:
        best = min(trained, key=lambda h: h.get("val_minade1", 1e9))
        gap = best.get("val_minade1", 0) - best.get("val_minade6", 0)
        ok = ranker_ok(best)
        print(
            f"RANKER done. best ADE1 {best.get('val_minade1', 0):.4f} ADE6 {best.get('val_minade6', 0):.4f} "
            f"gap {gap:.4f} Error {best.get('val_error_score', 0):.4f} pick_acc {best.get('val_pick_acc', 0):.3f}",
            flush=True,
        )
        with open(os.path.join(args.out_dir, "success.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "success": ok,
                    "gap": gap,
                    "rules": {"val_minade6_lt": 0.70, "val_pick_acc_gt": 0.7, "gap_lt": 0.15},
                    **{k: best.get(k) for k in best},
                },
                f,
                indent=2,
            )
        print(
            f"SUCCESS={ok} (ADE6<0.70 and pick_acc>0.7 and ADE1-ADE6<0.15)",
            flush=True,
        )


if __name__ == "__main__":
    main()
