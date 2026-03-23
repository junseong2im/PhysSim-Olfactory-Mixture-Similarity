# OlfaBind: Olfactory Competitive Binding Network를 활용한 혼합물 유사도 예측

## 개요

OlfaBind는 분자 지문(Molecular Fingerprint)을 천체역학 시뮬레이션에 매핑하여 후각 혼합물 간의 지각적 유사도를 예측하는 물리 기반 신경망이다. 기존의 그래프 신경망(GNN) 기반 접근법과 달리, OlfaBind는 분자를 질량, 위치, 속도를 가진 천체(Celestial Body)로 변환한 뒤 N-body 중력 시뮬레이션을 수행하고, 궤도 안정성 분석을 통해 혼합물의 물리적 임베딩을 추출한다.

본 저장소는 OlfaBind 모델의 아키텍처 설계, 학습 전략 탐색, 대조 학습(Contrastive Learning) 실험, 물리 기반 손실 함수 실험 등 총 9개 버전(v17~v25)에 걸친 체계적인 실험 기록을 포함한다.

## 아키텍처

![OlfaBind 아키텍처](figures/architecture.png)

### 파이프라인 구조

OlfaBind 시스템은 3개의 핵심 모듈로 구성된다.

### Module 1: InputHardwareLayer (입력 하드웨어 계층)

분자 지문을 물리 시뮬레이션에 적합한 내부 표현으로 변환한다.

1. Fingerprint Grid Mapping: 2048차원 Morgan Fingerprint를 8x16 격자맵으로 재구성
2. Sparse Activation: 각 격자 셀에 학습 가능한 활성화 함수를 적용하여 희소 활성 패턴(Constellation) 생성
3. Channel Transformation: 격자 패턴을 128차원 원자 벡터(d_atom=128)로 변환

이 과정을 통해 각 분자는 128차원 벡터로 인코딩된다. 혼합물은 최대 20개 분자의 집합으로 표현되며, 유효하지 않은 슬롯은 마스킹된다.

### Module 2: PhysicsProcessingEngine (물리 처리 엔진)

#### 2.1 ConstellationToCelestial (매핑)

128차원 원자 벡터를 물리량으로 변환한다:
- 질량 (Mass, 1D): Softplus 함수로 양수 보장, clamp(max=5.0)
- 위치 (Position, 3D): tanh() x 2.0으로 [-2, 2] 범위
- 속도 (Velocity, 3D): tanh() x 0.5로 [-0.5, 0.5] 범위

이 매핑은 단일 선형 계층(Linear)으로 구현되며, v24 실험에서 MLP로 교체 시 성능 저하가 관찰되어 단순 선형 매핑이 최적임을 확인하였다.

#### 2.2 GravitationalEngine (중력 엔진)

N-body 중력 시뮬레이션을 기울기 추적(gradient tracking)이 가능한 미분 가능 연산으로 구현한다.

- 학습 가능한 중력 상수: log_G (nn.Parameter)
- Verlet 적분: 위치와 속도를 시간 스텝별로 갱신
- 가속도 클램핑: 수치 안정성을 위한 accel_clamp=100.0
- 질량 감쇠: 시뮬레이션 진행에 따라 질량이 점진적으로 감소
- 궤적 기록: 전체 시간 스텝의 위치를 (B, T, N, 3) 텐서로 기록

시뮬레이션 파라미터는 n_steps=4, dt=0.05가 최적이며, T=8 이상에서는 수치 불안정이 증가한다.

#### 2.3 OrbitalStabilityEvaluator (궤도 안정성 평가기)

시뮬레이션 궤적으로부터 20차원 물리 임베딩을 추출한다.

추출되는 물리 특징:
- 궤도 안정성 점수 (1D): 스펙트럼 분석 기반
- 고유값 기반 스펙트럼 지문 (8D): 상호작용 행렬의 고유값
- 평균 질량, 속도, 각운동량 (3D)
- 운동/위치 에너지 비율 (2D)
- 궤적 변동성 지표 (6D): 분산, 쌍별 거리 등

### Module 3: Similarity Prediction (유사도 예측)

두 혼합물의 물리 임베딩 차이를 기반으로 유사도를 예측한다.

1. Projection: 20D -> 64D (Linear + LayerNorm + GELU)
2. Absolute Difference: |P_A - P_B|
3. Similarity Head: 64D -> 32D -> 1D (MLP + sigmoid)

출력값은 0~1 범위의 유사도 점수이며, 학습 시 MSE 손실 함수를 사용한다.

## 학습 과정

![학습 과정](figures/training_process.png)

### Multi-Restart Training

OlfaBind의 손실 지형(loss landscape)은 다수의 국소 최솟값을 포함하여 단일 학습에서 최적해에 도달하는 확률이 낮다. 이를 해결하기 위해 Multi-Restart Training을 도입하였다:

1. 동일한 데이터 분할에 대해 N개의 독립적인 학습을 수행 (각각 다른 랜덤 시드)
2. 각 학습에서 검증 Pearson r 기준 최우수 모델을 선택
3. N개의 후보 중 최고 성능 모델을 최종 모델로 채택

실험 결과:
- 3-restart: r = 0.680 (v18 baseline)
- 10-restart: r = 0.780 (v25 baseline, +14.7% 향상)

Restart 횟수의 증가가 SWA, Warmup, Cosine Decay 등의 학습 기법보다 더 큰 성능 향상을 가져온다는 것이 본 실험의 핵심 발견이다.

### 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| d_input (입력 차원) | 2048 | Morgan Fingerprint |
| d_atom (원자 벡터 차원) | 128 | |
| n_steps (시뮬레이션 스텝) | 4 | T=2~4 최적 |
| dt (시간 간격) | 0.05 | |
| Batch Size | 16 | |
| Learning Rate | 3e-4 | Adam optimizer |
| Weight Decay | 1e-4 | |
| Epochs | 60 | Early stopping (patience=15) |
| Scheduler | CosineAnnealingLR | |
| Grad Clip | 1.0 | max_norm |
| Restarts | 10 | (best selection) |

## 실험 결과

![실험 결과 비교](figures/experiment_results.png)

### 5-Seed x 5-Fold Cross-Validation (Snitz et al., 2013 데이터셋)

본 실험은 Snitz 2013 데이터셋(360 혼합물 쌍)을 사용하여 5개 시드 x 5-fold 교차 검증(총 25회)으로 평가하였다.

| 버전 | 방법론 | Pearson r | Std | 이전 대비 |
|------|--------|:---------:|:---:|:--------:|
| v25 baseline | 원본 구조 + 10-restart | 0.780 | 0.039 | +0.100 |
| v25-C | HP grid search (T, dt) | 0.732 | 0.038 | +0.052 |
| v25-A | +SWA, warmup, grad accum | 0.727 | 0.037 | +0.047 |
| v18 | 원본 구조 + 3-restart | 0.680 | - | 기준선 |
| v19 | +InfoNCE 대조 학습 | 0.594 | 0.085 | -0.086 |
| v25-B | +Bushdid 데이터 증강 | 0.565 | 0.073 | -0.115 |
| v21 | +6개 구조 개선 | 0.553 | 0.117 | -0.127 |
| v22 | +물리 기반 손실 (HNN, PINN) | 0.532 | 0.119 | -0.148 |
| v23 | +Multi-scale, Trajectory Attention | 0.520 | 0.112 | -0.160 |
| v24 | +MLP mapper, 32D embedding | 0.440 | 0.090 | -0.240 |
| v20 | +Triplet Margin Loss | 0.436 | 0.112 | -0.244 |

### 각 버전별 상세 설명

v17: OlfaBind의 최초 검증 실험. 물리 엔진의 기본 동작 확인 및 벤치마크 설정.

v18: T-sweep (시뮬레이션 길이 탐색)과 multi-restart 도입. T=4가 최적임을 확인하고, 3-restart로 r=0.680 달성. 이후 모든 실험의 baseline이 됨.

v19: SimCLR 스타일 InfoNCE 손실 함수를 추가하여 대조 학습을 시도. 같은 혼합물의 서로 다른 augmented view를 positive pair로, 다른 혼합물을 negative pair로 학습. 대조 학습 목표가 유사도 예측 목표와 상충하여 성능 하락.

v20: Triplet Margin Loss로 전환. 같은 분자의 두 augmented view를 anchor-positive로, in-batch random을 negative로 학습. 데이터 부족(360 쌍)과 과도한 margin으로 심한 성능 하락 발생.

v21: v20의 문제를 개선하기 위해 6개 기법 추가: 적응적 margin, hard negative mining, similarity-weighted loss, projection head 개선, layer-wise learning rate, validation-aware weighting. 일부 개선되었으나 여전히 baseline 이하.

v22: 대조 학습을 완전히 배제하고 물리 기반 손실 함수만 사용하는 접근법 시도. Hamiltonian Trajectory Matching (해밀턴 궤적 매칭), PINN Regularization (물리 정보 신경망 정규화), Spectral Matching (스펙트럼 매칭) 3종 적용. 높은 변동성이 관찰됨.

v23: 모델의 자유도를 극대화하면서 안정성도 강화하는 접근법. Multi-scale Simulation (T=2,4,8 동시 실행), Trajectory Attention (궤적 어텐션), 8D Latent Space (8차원 잠재 공간), 10-restart, SWA, Mixup 등 적용. 복잡도 증가가 오히려 불안정화를 유발.

v24: 내부 구조만 개선하는 보수적 접근. ConstellationToCelestial 매퍼를 2-layer MLP + LayerNorm + Residual로 교체, 물리 임베딩을 20D에서 32D로 확장. 단순 Linear 매퍼보다 성능이 악화되어, 원본 매퍼가 이미 최적임을 확인.

v25: 구조 고정, 학습/데이터/하이퍼파라미터 3가지 최적화를 병행 실험. Strategy A (학습 전략: SWA, warmup, grad accumulation), Strategy B (데이터 증강: Bushdid pseudo-similarity), Strategy C (HP search: T, dt grid). 최종적으로 원본 구조 + 원본 학습 + 10-restart가 최고 성능 달성.

### 핵심 발견

1. 물리 엔진의 아키텍처는 이미 최적에 가깝다. v19~v24에서 시도한 모든 구조 변경(대조 학습 모듈, 물리 손실 함수, MLP 매퍼, 궤적 어텐션 등)은 baseline 대비 성능을 저하시켰다.

2. Restart 횟수가 성능의 가장 큰 결정 인자다. 다른 학습 기법(SWA, warmup, cosine scheduling)의 기여는 미미했으며, 단순히 restart 횟수를 3에서 10으로 증가시킨 것만 으로 r이 0.680에서 0.780으로 14.7% 향상되었다.

3. 외부 데이터는 도움이 되지 않는다. Bushdid discrimination 데이터를 pseudo-similarity로 변환하여 추가 학습에 활용한 결과, 오히려 노이즈로 작용하여 성능이 저하(r=0.565)되었다.

4. 시뮬레이션 길이 T=4가 최적이다. T=2와 비슷한 성능을 보이지만, T=8 이상에서는 수치 불안정으로 인해 급격한 성능 하락이 발생한다.

## 저장소 구조

```
Perfume-Ai-research/
├── README.md                                     # 본 문서
├── models/                                       # OlfaBind 핵심 모듈
│   ├── olfabind_input.py                         # Module 1: 입력 하드웨어 계층
│   ├── olfabind_engine.py                        # Module 2: 물리 처리 엔진
│   ├── olfabind_contrastive.py                   # 대조 학습 모듈 (v19~v21)
│   ├── olfabind_pipeline.py                      # 전체 파이프라인 통합
│   └── olfabind_ghost.py                         # Ghost Molecule 증강 모듈
├── experiments/                                  # 실험 스크립트 (v17~v25)
│   ├── v17_olfabind_validation.py                # 최초 검증
│   ├── v18_olfabind_validation.py                # T-sweep + multi-restart
│   ├── v19_contrastive_validation.py             # InfoNCE 대조 학습
│   ├── v20_triplet_physics_validation.py         # Triplet Margin Loss
│   ├── v21_enhanced_triplet.py                   # 6개 구조 개선
│   ├── v22_physics_native.py                     # 물리 기반 손실 함수 (HNN, PINN)
│   ├── v23_freedom_stability.py                  # Multi-scale + 8D latent
│   ├── v24_internal_improvement.py               # 내부 매퍼 교체
│   └── v25_optimization_trio.py                  # 3가지 최적화 전략 비교
├── results/                                      # 실험 결과 JSON
│   ├── v17_olfabind_validation.json
│   ├── v18_olfabind_validation.json
│   ├── v19_contrastive_validation.json
│   ├── v20_triplet_physics_validation.json
│   ├── v21_enhanced_triplet.json
│   ├── v22_physics_native.json
│   └── v25_optimization_trio.json
└── figures/                                      # 논문 그림
    ├── architecture.png                          # 아키텍처 다이어그램
    ├── experiment_results.png                    # 실험 결과 비교
    └── training_process.png                      # 학습 과정 시각화
```

## 실행 환경

- Python 3.12
- PyTorch 2.x (CUDA 지원)
- RDKit (분자 지문 생성)
- NumPy, SciPy, Scikit-learn, Pandas

## 데이터셋

본 연구에서 사용한 데이터셋은 DREAM Olfaction Challenge의 혼합물 데이터이다:

- Snitz et al. (2013): 360 혼합물 쌍, 21가지 지각적 유사도 측정 (주 학습 데이터)
- Bushdid et al. (2014): 6,864 discrimination 데이터 (pseudo-similarity 변환 시도, 비효과적)
- Ravia et al. (2020): 771 intensity 측정 데이터 (참조용)

## 인용

본 연구에서 참조한 주요 논문:

- Snitz, K., et al. (2013). Predicting Odor Perceptual Similarity from Odor Structure. PLoS Computational Biology.
- Bushdid, C., et al. (2014). Humans Can Discriminate More than 1 Trillion Olfactory Stimuli. Science.
- Ravia, A., et al. (2020). A Measure of Smell Enables the Creation of Olfactory Metamers. Nature.
- Lee, B., et al. (2023). A Principal Odor Map Unifies Diverse Tasks in Human Olfaction. Science.
- Greydanus, S., et al. (2019). Hamiltonian Neural Networks. NeurIPS.
- Raissi, M., et al. (2019). Physics-Informed Neural Networks. Journal of Computational Physics.

## 라이선스

MIT License