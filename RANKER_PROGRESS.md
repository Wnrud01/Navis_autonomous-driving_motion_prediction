# 학습 진행 정리 — 무엇이 됐고, 왜 ADE1이 안 붙었는지

작성: 2026-08-31  
프로젝트: `E:\motion_prediction`  
목표 Error: `0.5 * (minADE1 + minADE6)` ≤ 0.6 (추론 < 100ms면 penalty 없음)  
전제: Error ≥ minADE6 이므로 **minADE6 자체가 0.6 아래**여야 하고, ADE1을 ADE6에 붙여야 Error가 내려간다.

---

## 한 줄 요약

| 단계 | 결과 | 진행? |
|---|---|---|
| V11 decoder 30ep | val ADE6 **0.6264**, ADE1 **1.2515**, Error **0.9389** | 됨. ADE6 바닥 ~0.62 |
| collate cache | train 215157 / val 24097 | 됨. GPU 병목 해소 |
| V12 decoder 30ep | val ADE6 **0.6239**, ADE1 **1.2391**, Error **0.9315** | 됨. ADE6는 V11과 거의 같음 |
| V11 ranker (평가 버그) | 로그 ADE6=ADE1=**3.35** | **가짜 숫자.** 세 지표를 한 텐서에 누적 |
| V11 ranker (수정 후 8ep) | ADE6 **0.6258** 고정, ADE1_rank **1.360**, pick_acc **0.43** | 평가는 정상. flatten 헤드가 logits보다 나쁨 |
| V11 residual ranker 8ep | ADE6 **0.6258**, ADE1_rank **1.257** (prior 1.251) | prior는 정상. **잔차가 logits를 못 이김** |

Decoder는 학습됐다. Ranker 평가는 고쳤다. 고르기는 아직 실패다.  
flatten 헤드 ADE1 1.36 > V11 logits 1.25. residual 헤드도 1.257로 prior(1.251)를 조금 깨기만 했다.

---

## 1. 학습 과정 (실제로 돌린 순서)

```
TFRecord  E:\motion_data\rideflux_91f_full\rideflux
          85126 files, 91 frames (past 10 + current + future 80 @ 10Hz)
    │
    ▼  data_tools/preprocess_85k_v2.py
v2 packs  prediction_pt_85k_v2
          train 229653 / val 25725
          hist [N,11,6] = x,y,yaw,vx,vy,valid   높이 제외
    │
    ▼  train_motion_prediction_v11.py   from-scratch, 30ep, bf16
V11 decoder   checkpoints/v11_ade6
              1.76M params, cumsum+refine, 8초 골 워프 없음
              aWTA Top-4→Top-2, uniform ADE, FDE 항 0, goal aux 0.15
    │
    ▼  data_tools/cache_collate_v10.py
cache fp16    prediction_pt_85k_v2_cache
              train 215157 / val 24097
    │
    ├─► train_ranker.py  (V11 freeze)     ★ 여기가 막힘
    │
    ▼  train_motion_prediction_v12.py    from-scratch on cache, 30ep
V12 decoder   checkpoints/v12_ade6
              4.5M, 순서 게이트 attn + type별 헤드
              ADE6 바닥은 V11과 동일
    │
    ▼  V12 ranker   아직 안 함 (V11 ranker 성공 조건 미달)
```

데이터/팩/캐시는 다시 뽑지 않았다. 문제는 2단계 고르기(ranker)다.

---

## 2. Decoder 로그 — 이쪽은 진행됨

### 2.1 V11 (`checkpoints/v11_ade6/training.log`)

설정: lr 3e-4, batch 32 scenes, max 24 targets, hidden 256, AMP bf16, 랜덤 초기화.  
epoch 19에서 `last.pth` resume (10h 제한으로 한 번 끊김).

```
Epoch 01/30 (1693.5s) | Train ADE6 1.184 ADE1 2.009 | Val ADE6 0.9050 ADE1 1.6706 FDE6 2.2351 Error 1.2878
Epoch 05/30 (1627.0s) | Train ADE6 0.734 ADE1 1.427 | Val ADE6 0.7254 ADE1 1.3873 FDE6 1.7253 Error 1.0563
Epoch 10/30 (1648.1s) | Train ADE6 0.679 ADE1 1.327 | Val ADE6 0.7276 ADE1 1.4273 FDE6 1.6990 Error 1.0775
Epoch 15/30 (1645.2s) | Train ADE6 0.647 ADE1 1.268 | Val ADE6 0.6668 ADE1 1.3500 FDE6 1.5777 Error 1.0084
Epoch 20/30 (1561.1s) | Train ADE6 0.622 ADE1 1.226 | Val ADE6 0.6458 ADE1 1.2909 FDE6 1.5226 Error 0.9683
Epoch 25/30 (1510.2s) | Train ADE6 0.602 ADE1 1.195 | Val ADE6 0.6326 ADE1 1.2526 FDE6 1.4855 Error 0.9426
Epoch 30/30 (1872.9s) | Train ADE6 0.593 ADE1 1.184 | Val ADE6 0.6264 ADE1 1.2515 FDE6 1.4786 Error 0.9389
```

캐시 없이 live collate → epoch ~27분, GPU util 낮음.

ADE6는 0.90 → 0.63으로 내려갔다. ADE1은 ~1.25에서 멈춘다.  
이유는 aWTA가 **궤적 모양**만 당기고, 고르기는 `argmax(mode_head)`이기 때문이다. Top-4→Top-2는 커버리지용이지 ranking supervision이 아니다.

### 2.2 V12 (`checkpoints/v12_ade6/training.log`)

같은 loss/aWTA. 캐시 사용 → epoch ~4–5분.

```
Epoch 01/30 (263.4s) | Train ADE6 1.176 ADE1 1.968 | Val ADE6 0.8735 ADE1 1.8607 FDE6 2.1246 Error 1.3671
Epoch 10/30 (240.3s) | Train ADE6 0.673 ADE1 1.308 | Val ADE6 0.6909 ADE1 1.3501 FDE6 1.6274 Error 1.0205
Epoch 20/30 (393.8s) | Train ADE6 0.609 ADE1 1.199 | Val ADE6 0.6307 ADE1 1.2749 FDE6 1.5098 Error 0.9528
Epoch 27/30 (304.8s) | Train ADE6 0.581 ADE1 1.160 | Val ADE6 0.6239 ADE1 1.2391 FDE6 1.4691 Error 0.9315   ← best
Epoch 30/30 (295.9s) | Train ADE6 0.576 ADE1 1.155 | Val ADE6 0.6240 ADE1 1.2395 FDE6 1.4713 Error 0.9317
```

순서 게이트 + type별 헤드는 ADE6 바닥을 **깨지 못했다** (0.626 → 0.624). Error도 0.939 → 0.932.  
점수 병목은 decoder 용량이 아니라 **6개 중 하나를 못 고르는 것** + ADE6 0.62 바닥이다.

---

## 3. Ranker 1차 실행 — 로그가 거짓이었다

경로: `checkpoints/v11_ranker/`, `checkpoints/v11_ranker_fp32/`

### 버그

```python
ade6 = ade1_base = ade1_rk = torch.zeros((), device=device)
```

파이썬에서 텐서 **하나**에 이름만 세 개다. 아래 `+=` 세 줄이 같은 값에 더해진다.

```python
ade6      += ade[ix, winner].sum()                 # minADE6
ade1_base += ade[ix, base_logits.argmax(-1)].sum() # V11 logits
ade1_rk   += ade[ix, rk_logits.argmax(-1)].sum()   # ranker
```

로그의 3.35는 궤적 붕괴가 아니라 **세 지표의 합**이다.

| 실제 (V11 val) | 합에 들어간 값 |
|---|---|
| minADE6 ≈ 0.626 | 그대로 |
| ADE1_base ≈ 1.252 | 그대로 |
| ADE1_rank ≈ 1.47 (거의 랜덤) | 그대로 |
| **합** | **0.626 + 1.252 + 1.47 ≈ 3.35** |

그래서 ADE6 = ADE1_base = ADE1_rank가 소수점까지 같고, `gap=0` → `SUCCESS=true`가 됐다.

### 1차 로그 (`checkpoints/v11_ranker/training.log`) — 쓰지 말 것

```
Ranker 01/8 (262.8s) | pick_acc 0.410 | Val ADE6 3.3526 ADE1_base 3.3526 ADE1_rank 3.3526 acc 0.415 Error 3.3526
Ranker 05/8 (212.1s) | pick_acc 0.427 | Val ADE6 3.2397 ADE1_base 3.2397 ADE1_rank 3.2397 acc 0.426 Error 3.2397
Ranker 08/8 (212.4s) | pick_acc 0.431 | Val ADE6 3.2611 ADE1_base 3.2611 ADE1_rank 3.2611 acc 0.427 Error 3.2611
```

에폭마다 3.35 → 3.24로 줄어 보인 것은 freeze가 풀린 게 아니라, ranker ADE1이 1.47 → 1.36 정도로만 내려간 **합**이다.  
autocast를 끈 fp32 실험이 실패한 이유도 이 버그가 forward와 무관해서다. 체크포인트·캐시는 처음부터 정상이었다.

---

## 4. Ranker 수정 후 8ep — 평가 정상, 고르기는 실패

경로: `checkpoints/v11_ranker_fix/`  
ckpt: `checkpoints/v11_ade6/best_minade6.pth`  
데이터: 같은 캐시, 재전처리 없음  
에폭당 ~110s

### 로그 (`checkpoints/v11_ranker_fix/training.log`)

```
Ranker 00/pre-train (untrained head) | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 5.0227 acc 0.252 Error 2.8243
Ranker 01/8 (128.0s) | pick_acc 0.415 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.4693 acc 0.419 Error 1.0475
Ranker 02/8 (112.2s) | pick_acc 0.424 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.4292 acc 0.423 Error 1.0275
Ranker 03/8 (108.9s) | pick_acc 0.427 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.4306 acc 0.426 Error 1.0282
Ranker 04/8 (113.5s) | pick_acc 0.429 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.3932 acc 0.428 Error 1.0095
Ranker 05/8 (115.2s) | pick_acc 0.431 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.3839 acc 0.430 Error 1.0049
Ranker 06/8 (109.9s) | pick_acc 0.433 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.3601 acc 0.432 Error 0.9930
Ranker 07/8 (113.9s) | pick_acc 0.434 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.4001 acc 0.431 Error 1.0130
Ranker 08/8 (109.9s) | pick_acc 0.435 | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.4004 acc 0.431 Error 1.0131
```

| | ADE6 | ADE1_base | ADE1_rank | pick_acc | Error | gap |
|---|---|---|---|---|---|---|
| 학습 전 (랜덤 헤드) | 0.6258 | 1.2511 | 5.023 | 0.252 | 2.824 | — |
| best ep 6 | **0.6258** | **1.2511** | **1.360** | **0.432** | 0.993 | 0.734 |
| ep 8 | 0.6258 | 1.2511 | 1.400 | 0.431 | 1.013 | 0.774 |
| 성공 기준 | < 0.70 | (참고) | ADE6+0.15 | > 0.70 | — | < 0.15 |

`success.json`: `"success": false`

학습 전 val이 ADE6 0.6258 / ADE1_base 1.2511 → **frozen V11 forward는 정상**.  
ADE6가 8에폭 내내 0.6258 → **decoder는 freeze 유지**.

---

## 5. 왜 고르기가 진행이 안 됐는지

평가 버그를 고친 뒤에도 pick_acc는 **0.415 → 0.435**에서 멈춘다. 랜덤 1/6 ≈ 0.167보다는 낫고, V11 logits를 이기지는 못한다.

### 5.1 헤드가 보는 정보가 부족하다

`src/hyp_ranker.py` (74,673 params):

```python
class HypRanker(nn.Module):
    def forward(self, traj, agent_tok, type_idx):
        te = self.traj_enc(traj.reshape(b, k, t * 2))          # 80×2 flatten
        ctx = self.ctx_enc(torch.cat([
            agent_tok[:, :16],                                  # speed/size/type 요약만
            self.type_emb(type_idx.clamp(0, 2)),
        ], dim=-1))
        ctx = ctx.unsqueeze(1).expand(b, k, -1)                 # 6모드에 같은 ctx
        return self.scorer(torch.cat([te, ctx], dim=-1)).squeeze(-1)
```

넣는 것: **6개 궤적 좌표 + agent 16차원 + type**.  
안 넣는 것: lane, lead, signal, map, neighbor, decoder hidden `h`, V11 `mode_head` logits.

V11의 `mode_head`는 fusion hidden 위에서 학습돼 ADE1 1.25를 만든다.  
Ranker는 그 context를 버리고 궤적 모양만 보고 다시 고른다. 그래서 ADE1_rank **1.36 > ADE1_base 1.25**.

### 5.2 ctx가 모드별로 다르지 않다

`agent_tok`과 `type_idx`는 타겟당 하나다. 6개 가설에 `expand`만 한다.  
모드 간 차이는 `traj_enc(flatten(x,y))`뿐이라, 비슷한 6개 궤적(aWTA Top-2)을 구분하기 어렵다.

### 5.3 라벨/로스가 약하다

- 라벨: ADE 최소 인덱스 하나. 2등과 차이가 작아도 hard CE.
- 80 step flatten은 초반 점이 후반을 묻는다. 사람이 고를 때 쓰는 분기(lane follow vs turn)가 희석된다.
- GT는 입력하지 않는 것은 맞다. 문제는 GT가 아니라 **장면 토큰을 안 주는 것**.

### 5.4 학습은 되고 있었다 — 천장만 낮다

loss/acc가 아주 안 움직인 건 아니다.

- train pick_acc 0.415 → 0.435
- val ADE1_rank 5.02 → 1.36 (랜덤 헤드에서 한 번 점프) → 이후 1.36~1.40 진동
- ep 6 이후 다시 나빠짐 (1.36 → 1.40)

즉 8에폭을 더 돌린다고 ADE1이 0.63으로 붙지 않는다. 헤드/입력 설계가 한계다.

### 5.5 성공 조건을 세 개로 나눈 이유

예전: `ADE1 - ADE6 < 0.15` 만 보면 둘 다 3.x여도 통과.  
지금:

1. `val_minade6 < 0.70` — 본학습 0.626의 1.1배. forward 오염 차단
2. `val_pick_acc > 0.70` — 실제로 승자를 고르는가
3. `val_minade1 - val_minade6 < 0.15` — ADE1이 ADE6에 붙었는가

현재는 1만 통과, 2·3 실패.

---

## 4.5 Residual ranker 8ep — prior는 정상, 잔차는 실패

경로: `checkpoints/v11_ranker_residual/`  
\(s_k = \ell_k^{\mathrm{V11}} + r_\theta(z_k)\). \(r_\theta\)만 학습.  
\(z_k\): 종점 / 4초 지점 / 평균 yaw / 횡방향 편차 + `agent_tok` / `lane_tok` / `inter_tok`. flatten 없음.  
로스: CE + hinge(\(m=0.5\)).

```
Ranker 00/pre-train (residual=0) | Val ADE6 0.6258 ADE1_base 1.2511 ADE1_rank 1.2511 acc 0.417 Error 0.9385
Ranker 01/8 (111.8s) | pick_acc 0.419 | ADE1_rank 1.2593 acc 0.416 Error 0.9425
Ranker 02/8 (95.9s)  | pick_acc 0.425 | ADE1_rank 1.2566 acc 0.427 Error 0.9412   ← best
Ranker 08/8 (96.9s)  | pick_acc 0.433 | ADE1_rank 1.2625 acc 0.430 Error 0.9442
```

학습 전 ADE1_rank = ADE1_base = **1.2511** → V11 logits가 들어갔고 residual 마지막 층 0 초기화가 맞다.  
이후 ADE1_rank는 한 번도 1.2511 아래로 안 내려갔다. pick_acc 0.417 → 0.43.

`SUCCESS=false`. 잔차가 prior를 이기지 못했다. lane/종점 요약만으로는 ADE-winner 신호가 부족하다.

---

## 6. 수정한 코드

파일: `train_ranker.py`  
변경점 세 가지: (1) 누적 텐서 분리 (2) 학습 전 val (3) 성공 조건 3개.

### 6.1 평가 누적 — 핵심 버그 수정

```python
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
        rk_logits = ranker(pred, samples["agent_tok"], samples["type_idx"])
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
```

### 6.2 학습 시작 전 val + 학습 루프 + 성공 판정

```python
    val0 = evaluate_ranker(model, ranker, val_loader, device, amp_dtype, args.max_val_steps or None)
    rec0 = {"epoch": 0, "train_loss": None, "train_pick_acc": None, "epoch_sec": 0.0, **val0}
    history.append(rec0)
    log_val("Ranker 00/pre-train (untrained head)", rec0)
    if val0.get("val_minade6", 99.0) >= 0.70:
        print("ABORT: frozen decoder val ADE6 too high. Check forward, not the ranker head.")
        return

    for epoch in range(1, args.epochs + 1):
        ranker.train()
        model.eval()
        for step, samples in enumerate(train_loader, start=1):
            samples = move_samples(samples, device)
            pred, _, _ = predict_hyps(model, samples, amp_dtype, device)
            ade = ade_per_mode(pred, samples["future"], samples["future_valid"])
            winner = ade.argmin(dim=-1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = ranker(pred.detach(), samples["agent_tok"], samples["type_idx"])
                loss = F.cross_entropy(logits, winner)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5.0)
            opt.step()
        val = evaluate_ranker(...)
        # save last.pth / best_ranker.pth on val_minade1

    trained = [h for h in history if h.get("epoch", 0) >= 1]
    best = min(trained, key=lambda h: h.get("val_minade1", 1e9))
    ok = ranker_ok(best)   # ADE6<0.70 and pick_acc>0.7 and ADE1-ADE6<0.15
```

전체 파일은 `E:\motion_prediction\train_ranker.py`.

랭커 헤드 `src/hyp_ranker.py`는 **수정하지 않았다.** 평가만 고쳤다. 고르기 성능 한계는 이 헤드 설계에 있다.

---

## 7. 다음에 손댈 곳 (아직 안 함)

V12 ranker는 돌리지 말 것. 같은 헤드면 같은 천장이다.

고르기를 살리려면 평가가 아니라 **입력을 바꿔야** 한다.

1. Ranker에 lane / lead / map / V11 hidden 또는 `mode_head` logits를 같이 넣기
2. 궤적은 flatten 대신 종점·중점·yaw, 또는 per-step encoder
3. V11 logits를 prior로 두고 residual ranking (지금 헤드가 logits보다 나쁨)
4. Hard CE 대신 margin / listwise (2등과 ADE 차이가 작음)

ADE6 0.62 바닥을 0.6 아래로 내리는 것은 ranker와 별개다. ranker는 ADE1만 당긴다.  
Error 하한은 지금 `0.5*(1.25+0.63)=0.94` 근처이고, 고르기가 완벽해도 `0.5*(0.63+0.63)=0.63`이라 목표 0.6에는 ADE6도 더 내려야 한다.

---

## 8. 파일 위치

| 내용 | 경로 |
|---|---|
| 이 문서 | `E:\motion_prediction\RANKER_PROGRESS.md` |
| 수정된 ranker 학습 | `E:\motion_prediction\train_ranker.py` |
| 랭커 헤드 (미수정) | `E:\motion_prediction\src\hyp_ranker.py` |
| V11 로그/가중치 | `E:\motion_prediction\checkpoints\v11_ade6\` |
| V12 로그/가중치 | `E:\motion_prediction\checkpoints\v12_ade6\` |
| ranker 가짜 로그 | `E:\motion_prediction\checkpoints\v11_ranker\` |
| ranker 수정 후 로그 | `E:\motion_prediction\checkpoints\v11_ranker_fix\` |
