# 🚗 Autonomous Driving AI Challenge: Motion Prediction Pipeline

자율주행 모션 예측(Motion Prediction) 트랙을 위한 **전체 파이프라인(원시 데이터 변환 $\rightarrow$ 전처리 캐시 생성 $\rightarrow$ 대규모 모델 학습 $\rightarrow$ NMS/SWA 후처리 평가 $\rightarrow$ 제출 파일 생성)** 통합 가이드입니다.

---

## 📁 0. 환경 구축 (Environment Setup)

### 1) 필수 사양
- **OS**: Windows 10/11 또는 Linux (Ubuntu 20.04/22.04)
- **Python**: 3.10 또는 3.11 권장
- **GPU**: NVIDIA GPU (RTX 3080 / 3090 / 4080 / 4090 / 5080 등, 16GB+ VRAM 권장)
- **CUDA**: CUDA 11.8 ~ 12.4+ 지원

### 2) 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 🔄 1단계: 원본 데이터(TFRecord) $\rightarrow$ `.pt` 변환

TFRecord 포맷의 원본 자율주행 데이터(과거 궤적, 미래 궤적, 도로 차선 폴리라인, 신호등 등)를 고속 PyTorch `.pt` 팩으로 변환합니다.

```bash
# Windows
python data_tools/preprocess_85k_v2.py --raw-root "E:/motion_prediction/data/raw" --out-root "data/processed/prediction_pt_85k_v2" --workers 12

# Linux
python data_tools/preprocess_85k_v2.py --raw-root "/path/to/raw_tfrecords" --out-root "data/processed/prediction_pt_85k_v2" --workers 16
```
- **출력 경로**: `data/processed/prediction_pt_85k_v2/{train,val}/*.pt`
- **주요 기능**:
  - 10Hz 8초 미래 궤적 (80스텝) 및 1초 과거 궤적 (11스텝) 추출
  - 16개 핵심 도로 차선 폴리라인 (20개 포인트 $\times$ 8차원 특성) 추출
  - 멀티프로세싱 병렬 처리 지원

---

## ⚡ 2단계: `.pt` 파일 $\rightarrow$ 고속 GPU Collate 캐시 변환

학습 중 CPU 병목을 0으로 만들고 GPU 100% 가동을 위해, 타깃 중심 좌표계 변환 및 텐서 패킹을 사전 연산하여 캐싱합니다.

```bash
python data_tools/cache_collate_v13.py --data-root "data/processed/prediction_pt_85k_v2" --out-dir "data/processed/prediction_pt_85k_v2_cache_v13" --workers 12
```
- **출력 경로**: `data/processed/prediction_pt_85k_v2_cache_v13/{train,val}/*.pt`
- **효과**: 1에폭 학습 속도가 3배 이상 빨라지며, 메모리 효율적인 `float16` 텐서로 압축 저장됩니다.

---

## 🧠 3단계: 대규모 모델 학습 (Large / X-Large Scale Training)

### 옵션 A: V16 Large 2단계 모델 (6.57M ~ 25.8M) — **[추천 / 최고 검증]**
- **특징**: Stage 1 다중 모드 제안기(Proposal) + Stage 2 국소 차선 밀착 정밀화기(Refinement) + 3단계 aWTA 심층 감독
- **학습 시간**: 약 2.5시간 (30 에폭)

```bash
python train_motion_prediction_v16.py \
    --cache-root "data/processed/prediction_pt_85k_v2_cache_v13" \
    --out-dir "checkpoints/v16_twostage" \
    --batch-scenes 32 \
    --epochs 30 \
    --lr 2e-4 \
    --hidden 256 \
    --amp bf16 \
    --workers 8
```

### 옵션 B: V17 X-Large 초대형 모델 (45.2M) — **[최대 표현력 & 차선/소셜 그래프]**
- **특징**: `hidden=768`, 12 헤드 Polyline Graph Self-Attention + Social Graph Self-Attention + 2단계 디코더
- **학습 시간**: 약 8.5시간 (30 에폭)

```bash
python train_motion_prediction_v17.py \
    --cache-root "data/processed/prediction_pt_85k_v2_cache_v13" \
    --out-dir "checkpoints/v17_xlarge" \
    --batch-scenes 24 \
    --epochs 30 \
    --lr 1.5e-4 \
    --hidden 768 \
    --nhead 12 \
    --dropout 0.1 \
    --amp bf16 \
    --workers 8
```

---

## 🎯 4단계: Test-Time NMS 후처리 & SWA 모델 앙상블 평가

학습된 모델의 가중치를 앙상블하거나, 추론 시 **밀도 가중 NMS (Soft Density NMS)**를 적용하여 재학습 없이 오차를 즉시 추가 단축시킵니다.

### 1) 단일 모델 NMS 클러스터링 평가
```bash
python evaluate_v16_nms.py \
    --ckpt "checkpoints/v16_twostage/best_error_score.pth" \
    --cache-root "data/processed/prediction_pt_85k_v2_cache_v13" \
    --batch-scenes 64
```

### 2) SWA 가중치 평균 (Model Soup) + NMS 평가
```bash
python evaluate_v16_swa.py \
    --ckpt-dir "checkpoints/v16_twostage" \
    --cache-root "data/processed/prediction_pt_85k_v2_cache_v13" \
    --batch-scenes 64
```

---

## 📦 5단계: 대회 제출용 예측 파일 생성 (Submission)

검증/테스트 데이터셋에 대해 최종 공식 포맷 제출 파일을 생성합니다.

```bash
python generate_test_public_submission.py \
    --ckpt "checkpoints/v16_twostage/best_error_score.pth" \
    --data-root "data/processed/prediction_pt_85k_v2" \
    --out-file "submission_v16_twostage.parquet"
```

---

## 📊 성능 지표 산출 공식 (Evaluation Metric)

$$\text{Error Score} = 0.5 \times (\text{minADE1} + \text{minADE6}) \times \left(1 + \frac{\max(0, T_{\text{infer}} - 100)}{200}\right)$$

- **minADE6**: 6개 모드 중 실제 정답과 가장 가까운 궤적의 평균 변위 오차 (m)
- **minADE1**: 모델이 1등으로 선택한 최고 확신도 궤적의 평균 변위 오차 (m)
- **$T_{\text{infer}}$**: 추론 지연 시간 (ms). **100ms 이내 시 속도 감점 0% (완전 면제)**.
