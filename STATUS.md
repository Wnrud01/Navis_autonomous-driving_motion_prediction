# Current status (2026-08-31)

Live tree: `E:\motion_prediction`  
Remote: `git@github.com:Wnrud01/Navis_autonomous-driving_motion_prediction.git`  
Raw TFRecords: `E:\motion_data\rideflux_91f_full` (not in git)

Target: Error = `0.5 * (minADE1 + minADE6)` ≤ 0.6, latency < 100 ms.

Full write-up: [RANKER_PROGRESS.md](RANKER_PROGRESS.md)

## Results

| Stage | val minADE6 | val minADE1 | Error | Note |
|---|---|---|---|---|
| V11 decoder (ep 30) | **0.6264** | **1.2515** | **0.9389** | mixed tokens, no 8s goal warp |
| V12 decoder (ep 27 best) | **0.6239** | **1.2391** | **0.9315** | ordered gated attn + type heads; ADE6 floor unchanged |
| Ranker eval bug | 3.35 (fake) | 3.35 (fake) | 3.35 | `ade6 = ade1_base = ade1_rk = zeros()` aliased one tensor |
| Ranker after eval fix | 0.6258 | 1.360 | 0.993 | flatten head worse than V11 logits 1.25 |
| Residual ranker (ep 2 best) | 0.6258 | 1.257 | 0.941 | \(s=\) V11 logits \(+ r_\theta\); residual never beat 1.251 |

Ranker success gate (all three): ADE6 < 0.70, pick_acc > 0.70, ADE1 − ADE6 < 0.15.  
Current: only ADE6 passes. pick_acc ~0.43.

## What is in this repo

- Preprocess v2, collate cache, V10–V12 trainers
- `train_ranker.py` + `src/hyp_ranker.py` (residual ranker: prior + traj summary + lane/inter, CE + hinge)
- Training logs under `checkpoints/*/training.log` (weights `*.pth` are gitignored)

## What is not solved

ADE1 is stuck near 1.25 (decoder logits). Ranker cannot pick the ADE-winner.  
Even a perfect ranker would give Error ≈ `0.5*(0.63+0.63)=0.63`, still above 0.6, so ADE6 must also drop.
