# V11 학습 방식과 Ranker 오류

이 zip은 **V11(minADE6) 학습 코드**와 **ADE1을 ADE6에 붙이려던 ranker 코드**입니다.  
랭커 실행 결과는 오류였습니다. 원인과 수정 방법을 아래에 적습니다.

---

## 1. V11이 하는 일 (1단계, 궤적)

- 입력: v2 팩 (hist 11×6, 차로 k/N, 신호, 이웃). 높이 0.
- 출력: 궤적 6개 + 골 6개 + logits 6개.
- 디코더: `cumsum(delta)` + refine. **8초 골에 궤적을 붙이지 않음.**
- 학습: aWTA. 정답과 가까운 모드만 당김. 초반 Top-4 → 후반 Top-2.
- loss: 전 구간 ADE 균등 (`time_weight_end=1`). FDE 항 없음. goal 보조 0.15.
- 가중치: **랜덤 초기화.** V9/V10 미사용.

최종 (epoch 30):

| | val |
|---|---|
| minADE6 | **0.626** |
| minADE1 | **1.252** |
| Error | **0.939** |

minADE1은 `argmax(logits)` 한 개의 ADE입니다. Top-4/2는 궤적 모양용이라 ADE1이 ADE6로 붙지 않습니다.

관련 파일:

- `train_motion_prediction_v11.py` — 학습 엔트리
- `src/train_motion_prediction_v11.py` — 워프 없는 디코더
- `src/train_motion_prediction_v10.py` — 토큰·collate (V11이 상속)
- `src/losses/awta_loss.py` — aWTA, last-True-valid FDE
- `train_motion_prediction_v3.py` — epoch 루프

---

## 2. Ranker가 하려던 일 (2단계, 고르기)

목표: **minADE1 ≈ minADE6** (항상 6개 중 GT에 제일 가까운 것을 고름).

올바른 프로토콜:

| | 학습 | 추론 |
|---|---|---|
| 입력 | 토큰 + **6개 궤적** | 토큰 + **6개 궤적** |
| 라벨 | `argmin_k ADE(pred_k, GT)` 인덱스 | 없음 |
| 출력 | 6개 점수 | `argmax` 1개 |

- V11 디코더는 **freeze**.
- GT는 **입력에 넣지 않음** (추론 때 없음).
- 관련 파일: `train_ranker.py`, `src/hyp_ranker.py`
- 캐시: `data_tools/cache_collate_v10.py`, `src/cached_collate.py`

성공 기준(코드): `ADE1 - ADE6 < 0.15`.

---

## 3. 무엇이 오류였나

실행 로그 (`checkpoints/v11_ranker/training.log`):

```
Val ADE6 3.24  ADE1_base 3.24  ADE1_rank 3.24  acc 0.43  Error 3.24
SUCCESS=True  gap=0.0
```

V11 본학습 val은 ADE6 **0.626** / ADE1 **1.252** 인데, 랭커 val은 둘 다 **3.24**입니다.

### 오류 코드 (수정 전 `predict_hyps`)

```python
@torch.no_grad()
def predict_hyps(model, samples, amp_dtype, device):
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
        pred, goals, logits = model_forward(model, samples)
    return pred.float(), goals.float(), logits.float()
```

랭커 기본 `--amp bf16`이라, **이미 학습이 끝난 V11**을 bf16 autocast로 다시 돌렸습니다.

같은 `best_minade6.pth`를 **fp32, autocast 없음**으로 캐시 val에 돌리면:

- minADE6 ≈ **0.63**
- minADE1 ≈ **1.28**

즉 체크포인트와 캐시는 정상이고, **랭커 안의 frozen forward만 궤적을 붕괴**시켰습니다. 6개 모드 ADE가 거의 같아져서 ADE1=ADE6=3.24, gap=0 → 코드가 성공으로 판정. 실제 고르기 정확도는 **43%** (랜덤 17%보다 조금 나을 뿐)라 ADE1을 0.63에 붙인 게 아닙니다.

### 왜 bf16에서 무너지나

V11 학습 때도 bf16을 썼습니다. 차이는:

1. 학습 때는 매 스텝 가중치가 fp32 마스터에서 갱신되고, val도 **같은 학습 루프의 autocast**로 0.626이 나옴.
2. 랭커는 모델을 `eval()` + `requires_grad=False` 후 **별도 스크립트**에서 autocast. LayerNorm/GRU/cumsum이 bf16으로 가면 80-step 궤적이 학습 때와 다른 스케일로 나옴.
3. 그 붕괴된 6개에 대해 “승자 인덱스”를 배우므로, 라벨 자체가 본학습의 ADE-winner가 아님.

추가로 `SUCCESS = (ADE1 - ADE6 < 0.15)` 만 봐서, **둘 다 3.24로 망가진 경우**를 성공으로 처리한 것도 버그입니다. ADE6가 본학습 값(0.63 근처)인지도 봐야 합니다.

---

## 4. 해결 방법

### (1) 얼린 V11 forward는 fp32, autocast 끔

현재 `train_ranker.py`에 반영됨:

```python
@torch.no_grad()
def predict_hyps(model, samples, amp_dtype, device):
    with torch.amp.autocast("cuda", enabled=False):
        pred, goals, logits = model_forward(model, samples)
    return pred.float(), goals.float(), logits.float()
```

랭커 헤드(`HypRanker`)만 bf16/fp32 아무거나 가능. 궤적 생성은 반드시 fp32.

### (2) 성공 판정 강화

예:

- `val_minade6`가 V11 본학습 ADE6의 **1.1배 이하** (예: < 0.70)
- `val_pick_acc` > 0.7
- `val_minade1 - val_minade6 < 0.15`

세 가지를 같이 만족해야 성공.

### (3) 다시 돌릴 때

캐시는 그대로 씁니다. 전처리 다시 필요 없음.

```text
python train_ranker.py --arch v11 --ckpt E:\motion_prediction\checkpoints\v11_ade6\best_minade6.pth --out-dir E:\motion_prediction\checkpoints\v11_ranker
```

첫 val 줄에서 ADE6가 **~0.63**, ADE1_base가 **~1.25**여야 정상입니다. 둘 다 3.x면 또 붕괴입니다.

정상 forward 위에서 pick_acc가 올라가고 ADE1_rank가 1.25 → 0.7 쪽으로 줄어야 ADE1≈ADE6입니다.

---

## 5. zip 안 파일

| 파일 | 역할 |
|---|---|
| `README_V11_RANKER.md` | 이 문서 |
| `train_motion_prediction_v11.py` | V11 학습 |
| `src/train_motion_prediction_v11.py` | V11 디코더 |
| `src/train_motion_prediction_v10.py` | 토큰/collate |
| `src/losses/awta_loss.py` | aWTA |
| `train_motion_prediction_v3.py` | epoch 루프 |
| `train_ranker.py` | 랭커 (fp32 수정 포함) |
| `src/hyp_ranker.py` | 6궤적 점수 헤드 |
| `src/cached_collate.py` | 캐시 로더 |
| `data_tools/cache_collate_v10.py` | 캐시 생성 |
| `v11_ranker_run/training.log` | 오류가 난 실행 로그 |
| `v11_ranker_run/success.json` | 잘못된 성공 판정 |
