# V6 Motion Prediction 개선 기록

## 현재 기준선

기존 `checkpoints/v6_temporal/eval_24.json` 기준 공식 error score는 **1.12597**이다. 구성은 `minADE_1=1.45807m`, `minADE_6=0.79387m`이며, 공식식 `0.5 * (minADE_1 + minADE_6)`와 일치한다. 따라서 현재 병목은 best-of-6보다 **logits.argmax로 선택되는 top-1 trajectory**이다.

## 적용한 코드 변경

`src/losses/awta_loss.py`에 공식 평가의 top-1 선택 규칙을 직접 반영하는 hard-winner classification loss를 추가했다. 각 샘플에서 ADE가 가장 낮은 mode를 winner로 지정하고 `cross_entropy(logits, winner)`를 계산하여, 추론 시 `argmax(logits)`가 실제 ADE winner를 선택하도록 학습 압력을 추가한다.

기존 soft classification loss의 기본 가중치는 `0.3 -> 0.8`로 높였고, 새 hard-winner 항의 기본 가중치는 `0.5`로 설정했다. 또한 최종 위치 오차가 8초 tail에 남아 있는 문제를 고려하여 `weight_fde` 기본값을 `0.25 -> 0.75`로 높였다. 해당 값들은 `train_motion_prediction_v6.py`의 `--weight-cls`, `--weight-hard-cls`, `--weight-fde` 옵션으로 조절할 수 있다.

## 검증 결과

수정 후 Python compile 검사를 통과했고, RTX 5080에서 기존 v6 checkpoint를 불러오는 짧은 smoke test를 완료했다. 100 train steps와 50 validation steps에서 프로세스가 정상 종료되었으며, 새 loss 항이 포함된 상태로 checkpoint가 저장되었다. 단, 이 smoke test는 전체 데이터의 일부만 사용한 것이므로 최종 성능 판단용이 아니다.

## 전체 재학습 권장 명령

```powershell
python train_motion_prediction_v6.py `
  --data-root "E:\motion_planning\data\processed\prediction_pt_85k" `
  --out-dir ".\checkpoints\v6_hardcls" `
  --resume-ckpt ".\checkpoints\v6_temporal\best_error_score.pth" `
  --epochs 20 `
  --batch-scenes 32 `
  --workers 8 `
  --prefetch 4 `
  --lr 1.0e-4 `
  --tau-cls 0.30 `
  --weight-fde 0.75 `
  --weight-cls 0.8 `
  --weight-hard-cls 0.5 `
  --amp bf16
```

평가 시에는 `best_error_score.pth`와 `best_minade6.pth`를 각각 평가하고, 반드시 `val_error_score`가 가장 낮은 checkpoint를 제출 후보로 선택해야 한다. 목표 0.5는 현재 기준선에서 큰 폭의 개선이 필요한 수준이므로, 이번 변경만으로 달성된 것으로 간주하면 안 되며 전체 재학습 결과를 기준으로 다음 조정을 판단해야 한다.

## 다음 실험 순서

첫 번째 재학습에서 `minADE_1`이 충분히 내려가지 않으면 `--weight-hard-cls 1.0`으로 올리고 `--tau-cls 0.20`을 시험한다. `minADE_6`이 악화되면 `--weight-fde 0.5`로 낮추고 `--time-weight-end 1.75`를 시험한다. 각 실험은 동일한 validation split에서 `minADE_1`, `minADE_6`, `minFDE_6`, error score를 함께 기록해야 한다.

또한 `evaluate_official_motion_prediction.py`는 V1 입력 파이프라인을 사용하므로 V6 checkpoint의 공식 비교에는 사용하지 않는다. V6는 V3/V6의 `evaluate_validation_v3` 경로로 평가해야 한다.
