#!/usr/bin/env python3
"""Foresight two-stage trainer on 85k packs.

Stage 1 (ranker): freeze 6 physics/lane hypotheses, learn linear scorer w,b.
Stage 2 (traj):   8 residual waypoints on those rollouts, joint scorer + TrajHead.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.foresight.dataset import ForesightCollate
from src.foresight.losses import ALIGN, L2, batched_ade, metric_loss, ranker_loss
from src.foresight.scorer import LinearScorer
from src.foresight.traj import N_WP, TrajHead
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples, query_gpu

TYPE_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}


def hard_metrics(ades: torch.Tensor, logits: torch.Tensor, pred_xy: torch.Tensor, future: torch.Tensor, valid: torch.Tensor):
    b = torch.arange(ades.size(0), device=ades.device)
    best6 = ades.argmin(dim=-1)
    top1 = logits.argmax(dim=-1)
    ade6 = ades[b, best6]
    ade1 = ades[b, top1]
    last = (valid.sum(dim=-1).long() - 1).clamp(min=0, max=79)
    disp = torch.linalg.vector_norm(pred_xy - future[:, None, :, :], dim=-1)
    fde6 = torch.gather(disp, 2, last[:, None, None].expand(-1, 6, 1)).squeeze(2)
    fde1 = fde6[b, top1]
    fde6b = fde6[b, best6]
    return dict(
        minade1=float(ade1.mean()),
        minade6=float(ade6.mean()),
        minfde1=float(fde1.mean()),
        minfde6=float(fde6b.mean()),
        n=int(ades.size(0)),
    )


@torch.no_grad()
def evaluate(model_scorer, model_head, loader, device, amp_dtype, max_steps=None):
    model_scorer.eval()
    if model_head is not None:
        model_head.eval()
    sums = {k: 0.0 for k in ("minade1", "minade6", "minfde1", "minfde6")}
    n = 0
    by = {name: {**{k: 0.0 for k in sums}, "n": 0} for name in TYPE_NAMES.values()}
    for step, samples in enumerate(loader, start=1):
        if max_steps and step > max_steps:
            break
        if samples is None or samples["features"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model_scorer(samples["features"], samples["heuristics"])
            paths = samples["paths"]
            if model_head is not None:
                paths = model_head.warp(paths, samples["features"])
            ades = batched_ade(paths, samples["future"], samples["future_valid"].float())
        m = hard_metrics(ades, logits, paths, samples["future"], samples["future_valid"])
        for k in sums:
            sums[k] += m[k] * m["n"]
        n += m["n"]
        types = samples["type_idx"].cpu()
        b = torch.arange(ades.size(0), device=ades.device)
        best6 = ades.argmin(dim=-1)
        top1 = logits.argmax(dim=-1)
        ade6 = ades[b, best6].float().cpu()
        ade1 = ades[b, top1].float().cpu()
        last = (samples["future_valid"].sum(dim=-1).long() - 1).clamp(min=0, max=79)
        disp = torch.linalg.vector_norm(paths - samples["future"][:, None, :, :], dim=-1)
        fde6 = torch.gather(disp, 2, last[:, None, None].expand(-1, 6, 1)).squeeze(2)
        fde1 = fde6[b, top1].float().cpu()
        fde6b = fde6[b, best6].float().cpu()
        for i in range(ades.size(0)):
            name = TYPE_NAMES.get(int(types[i]), "Vehicle")
            by[name]["n"] += 1
            by[name]["minade6"] += float(ade6[i])
            by[name]["minade1"] += float(ade1[i])
            by[name]["minfde6"] += float(fde6b[i])
            by[name]["minfde1"] += float(fde1[i])
    if n == 0:
        return {"val_minade1": 0, "val_minade6": 0, "val_minfde1": 0, "val_minfde6": 0, "val_error_score": 0, "n": 0, "by_type": {}}
    out = {f"val_{k}": sums[k] / n for k in sums}
    out["val_error_score"] = 0.5 * (out["val_minade1"] + out["val_minade6"])
    out["n"] = n
    out["by_type"] = {}
    for name, bucket in by.items():
        nn = max(1, bucket["n"])
        out["by_type"][name] = {
            "count": bucket["n"],
            "minade6": bucket["minade6"] / nn,
            "minade1": bucket["minade1"] / nn,
            "minfde6": bucket["minfde6"] / nn,
            "minfde1": bucket["minfde1"] / nn,
        }
    return out


def save_ckpt(path, scorer, head, extra):
    blob = {
        "w": scorer.w.detach().cpu().tolist(),
        "b": float(scorer.b.detach().cpu()),
        **extra,
    }
    if head is not None:
        blob["traj"] = {k: v.detach().cpu().tolist() for k, v in head.state_dict().items()}
        blob["n_wp"] = N_WP
    torch.save({"model_state": {
        **{f"scorer.{k}": v for k, v in scorer.state_dict().items()},
        **({f"head.{k}": v for k, v in head.state_dict().items()} if head is not None else {}),
    }, "json": blob, **extra}, path)
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in blob.items() if k != "logs"}, f, ensure_ascii=False, indent=2)


def load_scorer_ckpt(scorer, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    sw = {k.split("scorer.", 1)[-1]: v for k, v in state.items() if k.startswith("scorer.") or k in ("w", "b")}
    if "w" in state and "scorer.w" not in state:
        sw = {"w": state["w"], "b": state["b"]}
    scorer.load_state_dict(sw, strict=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    p.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\foresight")
    p.add_argument("--stage", choices=["ranker", "traj", "all"], default="all")
    p.add_argument("--ranker-ckpt", default="")
    p.add_argument("--ranker-epochs", type=int, default=8)
    p.add_argument("--traj-epochs", type=int, default=20)
    p.add_argument("--batch-scenes", type=int, default=32)
    p.add_argument("--lr-ranker", type=float, default=0.03)
    p.add_argument("--lr-traj", type=float, default=0.01)
    p.add_argument("--loss", choices=("soft-wta", "soft-ce", "wta"), default="soft-wta")
    p.add_argument("--max-targets", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--max-packs", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = p.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.amp != "fp32" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    os.makedirs(args.out_dir, exist_ok=True)
    train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
    val_paths = list_pt_paths(args.data_root, "val", args.max_packs)
    probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
    val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    pin = device.type == "cuda"
    train_loader = make_loader(
        train_ds, args.batch_scenes, args.workers, args.prefetch, True,
        ForesightCollate(args.max_targets, True), pin,
    )
    val_loader = make_loader(
        val_ds, args.batch_scenes, args.workers, args.prefetch, False,
        ForesightCollate(args.max_targets, False), pin,
    )

    print("=" * 80)
    print(" FORESIGHT MOTION RANKER — frozen 6-hypotheses + linear scorer + traj residual")
    print(f" Data:     {args.data_root}")
    print(f" Out:      {args.out_dir}")
    print(f" Stage:    {args.stage}  loss={args.loss}")
    print(f" Scenes:   train {len(train_ds.paths)}  val {len(val_ds.paths)}")
    print(f" Device:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80, flush=True)

    scorer = LinearScorer().to(device)
    head = None
    history = []

    def run_ranker():
        opt = torch.optim.SGD(scorer.parameters(), lr=args.lr_ranker)
        best = float("inf")
        for epoch in range(1, args.ranker_epochs + 1):
            scorer.train()
            t0 = time.time()
            loss_sum = acc_sum = gap_sum = 0.0
            n_b = 0
            log_n = 0
            log_loss = log_acc = 0.0
            last = t0
            for step, samples in enumerate(train_loader, start=1):
                if samples["features"].shape[0] == 0:
                    continue
                samples = move_samples(samples, device)
                opt.zero_grad(set_to_none=True)
                logits = scorer(samples["features"], samples["heuristics"])
                out = ranker_loss(logits, samples["ades"], kind=args.loss, weights=scorer.w, l2=L2, align=ALIGN)
                out["loss"].backward()
                opt.step()
                loss_sum += float(out["loss"].detach())
                acc_sum += float(out["acc"].detach())
                gap_sum += float(out["gap"].detach())
                n_b += 1
                log_n += 1
                log_loss += float(out["loss"].detach())
                log_acc += float(out["acc"].detach())
                every = 50 if step <= 200 else args.log_every
                if step % every == 0 and log_n:
                    now = time.time()
                    gpu = query_gpu()
                    print(
                        f" [Ranker {epoch:02d}/{args.ranker_epochs}] Step [{step:05d}/{len(train_loader)}] | "
                        f"Loss {log_loss/log_n:.4f} acc {100*log_acc/log_n:.1f}% | "
                        f"{every/max(1e-6, now-last):.1f} steps/s | GPU {gpu['util']:.0f}% {gpu['mem_mb']:.0f}MB",
                        flush=True,
                    )
                    log_n = 0
                    log_loss = log_acc = 0.0
                    last = now
            val = evaluate(scorer, None, val_loader, device, amp_dtype)
            rec = {
                "stage": "ranker", "epoch": epoch,
                "train_loss": loss_sum / max(n_b, 1),
                "train_acc": acc_sum / max(n_b, 1),
                "train_gap": gap_sum / max(n_b, 1),
                "epoch_sec": time.time() - t0,
                **{k: v for k, v in val.items() if k != "by_type"},
            }
            history.append(rec)
            print(
                f"Ranker {epoch:02d}/{args.ranker_epochs} ({rec['epoch_sec']:.1f}s) | "
                f"loss {rec['train_loss']:.4f} acc {100*rec['train_acc']:.1f}% gap {rec['train_gap']:.3f} | "
                f"val ADE1 {val['val_minade1']:.4f} ADE6 {val['val_minade6']:.4f} Error {val['val_error_score']:.4f}",
                flush=True,
            )
            with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            if val["val_minade1"] < best:
                best = val["val_minade1"]
                save_ckpt(os.path.join(args.out_dir, "best_ranker.pth"), scorer, None, rec)
                print(f"[NEW BEST ranker minADE1] {best:.4f}", flush=True)
        save_ckpt(os.path.join(args.out_dir, "last_ranker.pth"), scorer, None, {"stage": "ranker"})
        return best

    def run_traj():
        nonlocal head
        head = TrajHead().to(device)
        opt = torch.optim.Adam(list(scorer.parameters()) + list(head.parameters()), lr=args.lr_traj, weight_decay=1e-4)
        best = float("inf")
        for epoch in range(1, args.traj_epochs + 1):
            scorer.train()
            head.train()
            t0 = time.time()
            loss_sum = a1_sum = a6_sum = 0.0
            n_b = 0
            log_n = 0
            log_loss = log_a6 = log_a1 = 0.0
            last = t0
            for step, samples in enumerate(train_loader, start=1):
                if samples["features"].shape[0] == 0:
                    continue
                samples = move_samples(samples, device)
                opt.zero_grad(set_to_none=True)
                logits = scorer(samples["features"], samples["heuristics"])
                pred_xy = head.warp(samples["paths"], samples["features"])
                ades = batched_ade(pred_xy, samples["future"], samples["future_valid"].float())
                metrics = metric_loss(ades, logits)
                rank = ranker_loss(logits, ades.detach(), kind="soft-wta", weights=scorer.w, l2=L2, align=ALIGN)
                k_star = ades.argmin(-1)
                b = torch.arange(ades.size(0), device=ades.device)
                wta_reg = ades[b, k_star].mean()
                loss = metrics["ade1"] + metrics["ade6"] + wta_reg + 0.3 * rank["loss"]
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach())
                a1_sum += float(metrics["ade1"].detach())
                a6_sum += float(metrics["ade6"].detach())
                n_b += 1
                log_n += 1
                log_loss += float(loss.detach())
                log_a1 += float(ades[b, logits.argmax(-1)].mean().detach())
                log_a6 += float(ades.min(-1).values.mean().detach())
                every = 50 if step <= 200 else args.log_every
                if step % every == 0 and log_n:
                    now = time.time()
                    gpu = query_gpu()
                    print(
                        f" [Traj {epoch:02d}/{args.traj_epochs}] Step [{step:05d}/{len(train_loader)}] | "
                        f"Loss {log_loss/log_n:.4f} ADE6 {log_a6/log_n:.3f} ADE1 {log_a1/log_n:.3f} | "
                        f"{every/max(1e-6, now-last):.1f} steps/s | GPU {gpu['util']:.0f}% {gpu['mem_mb']:.0f}MB",
                        flush=True,
                    )
                    log_n = 0
                    log_loss = log_a6 = log_a1 = 0.0
                    last = now
            val = evaluate(scorer, head, val_loader, device, amp_dtype)
            rec = {
                "stage": "traj", "epoch": epoch,
                "train_loss": loss_sum / max(n_b, 1),
                "epoch_sec": time.time() - t0,
                **{k: v for k, v in val.items() if k != "by_type"},
            }
            history.append(rec)
            print(
                f"Traj {epoch:02d}/{args.traj_epochs} ({rec['epoch_sec']:.1f}s) | "
                f"loss {rec['train_loss']:.4f} | "
                f"val ADE1 {val['val_minade1']:.4f} ADE6 {val['val_minade6']:.4f} "
                f"FDE6 {val['val_minfde6']:.4f} Error {val['val_error_score']:.4f}",
                flush=True,
            )
            with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.out_dir, "last_type_breakdown.json"), "w", encoding="utf-8") as f:
                json.dump(val.get("by_type", {}), f, ensure_ascii=False, indent=2)
            if val["val_error_score"] < best:
                best = val["val_error_score"]
                save_ckpt(os.path.join(args.out_dir, "best_error_score.pth"), scorer, head, rec)
                print(f"[NEW BEST Error Score] {best:.4f}", flush=True)
            if val["val_minade6"] <= min((h.get("val_minade6", 1e9) for h in history if h.get("stage") == "traj"), default=1e9):
                save_ckpt(os.path.join(args.out_dir, "best_minade6.pth"), scorer, head, rec)
        save_ckpt(os.path.join(args.out_dir, "last.pth"), scorer, head, {"stage": "traj"})
        return best

    if args.stage in ("ranker", "all"):
        run_ranker()
    elif args.ranker_ckpt:
        load_scorer_ckpt(scorer, args.ranker_ckpt, device)
        print(f"-> loaded ranker {args.ranker_ckpt}", flush=True)

    if args.stage in ("traj", "all"):
        if args.stage == "all":
            rk = os.path.join(args.out_dir, "best_ranker.pth")
            if os.path.isfile(rk):
                load_scorer_ckpt(scorer, rk, device)
                print(f"-> resume ranker from {rk}", flush=True)
        elif args.ranker_ckpt:
            load_scorer_ckpt(scorer, args.ranker_ckpt, device)
        run_traj()

    print("Foresight training finished.", flush=True)


if __name__ == "__main__":
    main()
