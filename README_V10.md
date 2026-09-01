# V10 전처리·학습 — 수정 파일과 이유

제주 TFRecord 검증에서 발견한 문제를 고친 파일만 모았습니다.  
팩은 `E:\motion_prediction\data\processed\prediction_pt_85k_v2` 입니다. 높이 클램프·k/N은 collate에서 계산하므로 **이미 뽑은 팩을 다시 만들 필요는 없습니다.**

학습 엔트리: `python train_motion_prediction_v10.py`  
데이터 기본 경로: `E:\motion_prediction\data\processed\prediction_pt_85k_v2`  
원본 TFRecord: `E:\motion_data\rideflux_91f_full\rideflux`

---

## 파일별 수정 이유

### 1. `data_tools/preprocess_85k_v2.py`

| 문제 | 수정 |
|---|---|
| `agent_size_m`가 전부 0 | `state/length`는 없음. `state/current/{length,width,height}` 사용. current가 0이면 past 마지막 → future 첫 프레임 |
| 신호 xyz 없음 | `traffic_light_state/{past,current}/{x,y,state,valid,id}` → `tl_xy (L,11,2)`. future TL 없음 |
| hist 차원 | `[N,11,6] = x,y,yaw,vx,vy,valid` |
| 타깃 행 | 현재 valid인 객체 전부 + 에고를 팩에 유지. loss 필터는 collate에서 |

### 2. `src/map_polylines.py`

| 문제 | 수정 |
|---|---|
| `(id,type)` 묶은 뒤 **등장 순서**로 resample | 같은 id가 파일에 섞여 점 간격 137 m 지그재그, k/N 붕괴 |
| | 그룹 후 **평균 dir 투영으로 정렬**, dir이 죽으면 PCA. 그다음 20점 resample |
| 차로/맵 타입 | `LANE_TYPES=(1,2)`, 횡단 18 (19는 과속방지턱), 경계 15/16, 정지선 17 |

### 3. `src/lane_index.py`

k/N은 팩에 숫자로 안 넣고 collate에서 계산합니다.

| 문제 | 수정 |
|---|---|
| 왼쪽 끝 클러스터를 무조건 채택 | 에고에 가장 가까운 클러스터 `argmin(\|lat\|)`를 기준으로 좌우 확장 |
| 합성 `-20/-4/0/+4` → n=1, lat=-20 | 평행도로. 지금은 **n=3, idx=2, lat≈0** |
| 1.4–2.5 m에서 `break` | 같은 차로 조각인데 그보다 왼쪽 차로가 잘림. 합성 `0/2.0/5.5`가 n=1이 됨 |
| | `gap < 2.5` → **continue** (포인터 유지), `2.5–6.5` 채택, `> 6.5` break |
| `gap`을 건너뛴 점 기준으로 잼 | `0/2.0/8.0`에서 `8-2=6`이면 8 m를 옆 차로로 넣음. **마지막 채택 차로**와의 간격만 사용 (`8-0=8` → break, n=1) |

### 4. `src/train_motion_prediction_v10.py`

V10 모델·collate (AGENT / LANE k/N / MAP / INTER).

| 문제 | 수정 |
|---|---|
| 이웃 채널 8번 **height ≈ 3e34** | TFRecord 일부 `state/current/height` 쓰레기 값. bf16 Linear가 Inf→NaN. 배치 평균 loss가 NaN |
| | 길이 0–30 m, 너비 0–5 m 클램프. **높이는 0** (AGENT도 원래 길이·너비만 사용) |
| MAP 토큰 `NaN * 0 = NaN` | 신호/맵 miss일 때 `hit=False`여도 NaN이 남음. `torch.where`로 대체 |
| 팩에 NaN/Inf | collate 끝에서 float 텐서 `nan_to_num` |

### 5. `src/train_motion_prediction_v3.py`

| 문제 | 수정 |
|---|---|
| 맵 포인트 타입 19 (과속방지턱) | `encode_map_points(..., keep_types=KEEP_TYPES)`로 18 횡단 등 V10 KEEP 사용 |
| 이웃 크기에 height 포함 | `n_size` 클램프 후 **height 채널 0** |

### 6. `src/losses/awta_loss.py`

FDE/goal은 `valid.sum()-1`이 아니라 **마지막 True valid 프레임**. prefix가 아닌 마스크에서 인덱스가 어긋나지 않게.

### 7. `train_motion_prediction_v10.py`

V10 학습 엔트리. v2 팩, from-scratch, aWTA tau 1.5→0.25, Error = `0.5*(minADE1+minADE6)`. V9 가중치 미로드. FDE도 last-True-valid.

### 8. `train_motion_prediction_v3.py`

학습 루프 가드 (원인 수정 후에도 남은 폭주 배치 대비):

- loss가 NaN/Inf → 그 배치만 backward 안 함
- grad가 NaN/Inf → `optimizer.step()` 안 함 (가중치 전체 NaN 방지)

원인 자체는 height 3e34이며, 가드만으로는 해결되지 않았습니다.

### 9. `smoke_test_v10_pipeline.py`

회귀 테스트: 크기≠0, 폴리라인 정렬, 평행도로 `-20/-4/0/4`, dead-zone `0/2.0/5.5`, `0/2.0/8.0` 간격은 마지막 채택 차로 기준.

---

## NaN skip이 났던 이유 (학습)

예외 코드/트레이스백 없음. IEEE NaN.

- 입력 `target_hist`·토큰은 유한
- `neighbors[..., 8]` (height) max **3.38e34**
- bf16 Linear overflow → 일부 타깃 `pred` NaN (예: 17/374) → 배치 평균 loss NaN → skip

해결: 높이 0 + 길이/너비 클램프. 진단 200배치 NaN 0건.

---

## 다시 안 고쳐도 되는 것 (검증에서 통과)

- 신호 `x,y` past+current 11스텝, future 없음
- hist `[N,11,6]`
- 주행 차로 `{1,2}`, 횡단 18, 경계 16
- 팩에 현재 valid 전부 + 에고. `tracks_to_predict` 미사용
- k/N은 팩에 없음, collate에서 계산
- 미래는 라벨만
- aWTA last-True-valid FDE, minADE1=argmax logits, minADE6=6개 중 min
- Error `0.5×(ADE1+ADE6)`
- INTER 앞차 = neighbor 채널 12 `is_lead`
