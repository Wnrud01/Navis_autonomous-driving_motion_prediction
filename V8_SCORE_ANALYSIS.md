# V8 현재 점수 및 목표 분석

## 현재 상태

V8 학습 프로세스는 계속 실행 중이며, 현재 로그 기준으로 **Epoch 16/20, Step 7,000/7,177**까지 진행되었다. Epoch 16 validation 결과는 아직 기록되지 않았고, 마지막 완료 validation은 Epoch 15이다.

| 지표 | V8 Epoch 15 validation | 해석 |
|---|---:|---|
| minADE1 | 1.4622 m | 공식 top-1 병목 |
| minADE6 | 0.8139 m | best-of-6 성능 |
| minFDE6 | 13.2019 m | 장기 endpoint 오차가 여전히 큼 |
| error score | **1.1380** | `0.5 × (minADE1 + minADE6)` |

현재까지 V8의 가장 좋은 validation error score는 Epoch 1의 1.1661보다 개선된 **1.1380**이다. 하지만 기존 V6의 best score 1.1241보다 아직 낮지 않으며, Epoch 11 이후에는 1.14~1.18 범위에서 정체되어 있다.

## 완료 시점 점수 추정

현재 추세와 cosine learning-rate schedule을 고려하면 남은 4개 epoch에서 큰 폭의 추가 개선이 발생할 가능성은 낮다. 가장 합리적인 추정치는 다음과 같다.

| 전망 | 완료 시 예상 error score | minADE1 예상 | minADE6 예상 |
|---|---:|---:|---:|
| 보수적 | 1.15~1.18 | 1.47~1.53 m | 0.81~0.85 m |
| 중심 추정 | **1.13~1.15** | 1.44~1.49 m | 0.80~0.83 m |
| 낙관적 | 1.10~1.13 | 1.39~1.45 m | 0.78~0.82 m |

따라서 현 V8 학습을 끝까지 완료해도 **0.5에 도달할 가능성은 매우 낮고**, 최종 score는 약 **1.13~1.15**로 보는 것이 타당하다. GIF에서 개별 sample의 minADE6가 0.65m로 보이는 것은 전체 validation 평균이 아니므로 전체 점수 예측에 직접 사용하면 안 된다.

## error score 0.5에 필요한 조건

공식식은 `error score = 0.5 × (minADE1 + minADE6)`이다. 따라서 0.5를 달성하려면 두 지표의 합이 정확히 1.0 이하가 되어야 한다.

| 가정 | 필요한 조건 |
|---|---:|
| 현재 minADE6 0.8139m 유지 | minADE1이 **0.1861m 이하**여야 함 |
| 현재 minADE1 1.4622m 유지 | minADE6가 음수가 되어야 하므로 불가능 |
| 균형 목표 | minADE1 약 0.50m, minADE6 약 0.50m |
| minADE6를 0.40m까지 개선 | minADE1이 **0.60m 이하**여야 함 |
| minADE1을 0.70m까지 개선 | minADE6가 **0.30m 이하**여야 함 |

현재 Epoch 15 기준으로 균형 목표인 0.50m씩을 적용해도 minADE1은 약 66%, minADE6는 약 39% 감소해야 한다. 이는 단순히 epoch을 더 돌리거나 loss weight를 미세 조정해서 달성할 수 있는 차이가 아니라, **예측 표현과 mode 선택 구조를 함께 바꿔야 하는 수준**이다.

## 0.5를 목표로 할 때 필요한 핵심 개선

첫째, 현재 top-1 mode 선택은 `logits.argmax()`인데, mode score가 scene-specific trajectory quality를 충분히 반영하지 못한다. 각 mode의 trajectory와 context를 함께 입력받는 **per-mode score head** 또는 predicted ADE proxy를 도입하고, 학습 중 실제 ADE winner와 score ranking을 직접 맞춰야 한다. 현재 hard-winner classification 항을 추가했지만, trajectory가 생성된 이후의 mode ranking을 평가하는 구조가 아니므로 top-1 병목을 완전히 해결하지 못한다.

둘째, 8초 endpoint 오차가 `minFDE6≈13.2m`으로 매우 크다. goal loss와 FDE loss만 높이는 것보다 1초, 2초, 4초, 6초, 8초의 anchor point를 별도 supervision하고, 각 mode에 대해 endpoint와 전체 path를 동시에 회귀하는 multi-horizon loss가 필요하다. 특히 후반부가 발산하는 샘플을 별도 분석해야 한다.

셋째, mode diversity가 단순히 goal 간 거리를 벌리는 방식이면 정확한 mode를 만들기보다 서로 다른 mode를 강제로 멀리 보낼 수 있다. 실제 데이터의 turn/straight/stop 등 미래 분포를 기준으로 mode assignment를 구성하고, 불필요한 diversity penalty는 줄이는 것이 안전하다.

넷째, 목표가 실제로 0.5인지 확인하려면 단위와 평가 파이프라인을 먼저 감사해야 한다. V8 validation은 V3/V6 collate 경로를 사용해야 하며, V1 기반 `evaluate_official_motion_prediction.py`와 혼용하면 안 된다. 또한 future timestep 간격, 마지막 valid timestep, world-to-local 좌표 복원, 차량·보행자·자전거별 score를 분리해 확인해야 한다.

## 결론

현재 V8은 정상 학습 중이지만, 마지막 validation score 1.1380과 최근 정체 추세를 보면 완료 시점 예상 score는 **1.13~1.15**다. **0.5를 목표로 한다면 현재 V8 완료 후 추가로 per-mode scene-conditioned ranking, multi-horizon endpoint supervision, mode assignment 재설계, 데이터·평가 단위 감사가 필요하다.** 현재 구조에서 추가 epoch만으로는 목표 차이를 줄이기 어렵다.
