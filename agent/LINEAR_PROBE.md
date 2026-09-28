# MotionJEPA 100STYLE Linear-Probe Handoff

Collapse and held-out JEPA diagnostics that run alongside this probe are
specified separately in `agent/ONLINE_REPRESENTATION_METRICS.md`.

이 문서는 100STYLE 데이터 구성, MotionJEPA linear probing, raw-motion
baseline, label-shuffled control 및 representation-quality 해석에 관해 합의한
현재 상태를 정리한다. 과거의 window-level 80/10/10 split 및 validation-selected
결과는 현재 protocol과 호환되지 않으며 모델 비교에 사용하지 않는다.

## 문제의식

Random-initialized MotionJEPA encoder에서도 100STYLE style classification
top-1이 약 50%로 관찰되었다. 100-class chance accuracy가 약 1%라는 점을
고려하면 매우 높은 수치이므로 다음 가능성을 검토했다.

- 동일하거나 유사한 source window가 train/test에 들어간 leakage
- style별 recording 또는 preprocessing artifact
- raw motion 자체가 style에 대해 쉽게 분리되는 데이터 특성
- Transformer가 만드는 고차원 random-feature mapping
- pretrained representation 또는 probe pooling의 문제

여기서 source motion/recording은 하나의 BVH 파일을 뜻한다. 100STYLE에서는
각 source가 하나의 content를 담는다. 예를 들어 `BR`은 backward-right,
`FR`은 forward-right이고, 한 style은 content별 BVH를 가진다.

## 현재 데이터 protocol

Dataset은 `dataset/100style-soma77-processed`이다.

- 공식 `dataset/Frame_Cuts.csv`를 사용한다.
- 원본 약 60 FPS sequence를 `[START:STOP]` stop-exclusive로 먼저 자른다.
- Trim 이후 30 FPS fixed-step resampling을 수행한다.
- 90-frame non-overlapping window만 만들고 remainder는 버린다.
- Window canonicalization은 각 window에 독립적으로 적용한다.
- 기본 content는 `BR BW FR FW SR SW`이다.
- Train content는 `BR/BW/FR/SR/SW`이고 test content는 `FW`이다.
- Validation은 사용하지 않으며 `val.txt`와 manifest만 빈 상태로 유지한다.
- `ID`, `TR1`, `TR2`, `TR3`는 주 classification dataset에서 제외한다.
- Normalization mean/std는 trimmed train content만으로 계산한다.

실제 dataset 통계는 다음과 같다.

| Split | Contents | Sources | Windows |
|---|---|---:|---:|
| Train | BR, BW, FR, SR, SW | 500 | 13,144 |
| Validation | 없음 | 0 | 0 |
| Test | FW | 100 | 4,266 |

100개 style은 train과 test 양쪽에 모두 존재한다. 전체 원본은 810개 BVH이고,
기본 6개 content에 해당하는 600개 BVH의 파일명, FPS, frame-cut 범위를
검증했으며 오류는 없었다. 각 content에는 style별 source가 정확히 100개 있다.

Preprocessor는 다음을 strict하게 검증한다.

- content 이름, 중복 및 test-content 포함 여부
- content별 완전한 style/source 집합
- CSV column, 중복 style, `N/A`, 정수 및 범위 오류
- cut stop이 실제 BVH frame count를 넘지 않는지
- source FPS가 target FPS와 정확히 호환되는지

`meta.json`에는 content split, `validation_enabled: false`, frame-cut SHA256,
endpoint convention 및 trim-before-resample 여부가 기록된다. 이 metadata가
맞지 않는 기존 output은 자동 재사용하지 않는다.

## Offline MotionJEPA linear probe

기본 probe는 EMA target encoder를 frozen eval mode로 사용한다. Encoder에는
gradient가 생기지 않으며 optimizer에는 새 linear head parameter만 들어간다.
기본 global-mean probe는 valid output token만 평균낸 뒤 biased
`Linear(feature_dim, 100)`을 학습한다.

Validation-free protocol은 사전에 고정되어 있다.

- 50 epochs
- SGD
- Initial LR 0.3
- Momentum 0.9
- Weight decay 0
- Cosine schedule
- Selection: `fixed_last_epoch`

매 epoch validation 평가는 하지 않는다. 마지막 epoch head를
`linear-probe-final.pth.tar`로 저장한 뒤 FW test를 정확히 한 번 평가한다.
Summary에는 다음이 기록된다.

```text
selection: fixed_last_epoch
validation_used: false
test_content: FW
```

Validation 없는 dataset에서는 LR sweep과 validation-selected report를
명시적으로 거부한다. FW 결과를 보고 LR, probe epoch 또는 pretraining
checkpoint를 바꾸지 않는다.

한 checkpoint를 평가하는 명령은 다음과 같다.

```bash
python -m experiment.linear_probe \
  --checkpoint output/<run>/<checkpoint>-latest.pth.tar \
  --dataset-root dataset/100style-soma77-processed
```

## Online linear probe

Online probe도 매 cadence마다 현재 EMA target encoder를 frozen 상태로 평가하고
새 linear head만 학습한다.

- Encoder, predictor, EMA 및 pretraining optimizer는 업데이트하지 않는다.
- Probe 종료 후 head는 pretraining에 반영하지 않는다.
- Probe 전후 Python, NumPy, PyTorch RNG 상태를 복원한다.
- `linear_probe/test_*` trajectory만 diagnostic으로 기록한다.
- FW 성능으로 `<write_tag>-best-accuracy.pth.tar`를 만들지 않는다.
- 최종 pretraining model은 latest 또는 사전 지정 epoch checkpoint로 선택한다.

즉 online probe는 representation 관찰 장치이지 pretraining model-selection
criterion이 아니다.

## Normalized raw-motion linear baselines

Raw baseline은 CNN/Transformer와 같은 dataset loader, normalization,
optimizer, schedule, metrics 및 checkpointing 경로를 사용한다. 현재 단일-layer
baseline은 valid frame의 temporal mean을 구한 뒤 한 affine layer만 적용한다.

```text
normalized raw motion [B, 90, 366]
→ valid-frame temporal mean [B, 366]
→ Linear(366, 100)
```

Parameter 수는 36,700개이다. 학습 설정은 다음과 같다.

- 100 epochs
- AdamW, LR `3e-4`, weight decay `0.05`
- 5 warmup epochs 후 cosine decay to `1e-6`
- Batch size 256
- Gradient clipping 1.0
- BF16, seed 42
- Validation 없이 epoch 100 고정 선택 후 FW test 1회

실행:

```bash
python -m experiment.linear_probe.linear \
  --dataset-root dataset/100style-soma77-processed \
  --device cuda:0 \
  --overwrite
```

현재 temporal-mean raw 결과:

| Metric | Value |
|---|---:|
| FW top-1 | 80.08% |
| FW macro accuracy | 79.91% |
| FW top-5 | 95.57% |
| FW loss | 0.8726 |

이 결과는 시간 순서를 모두 버리고 평균 366-D pose/motion statistics만 사용해도
style이 매우 쉽게 분리됨을 보여준다.

이전 flatten baseline은 `[90,366]` 전체를 펼쳐 `Linear(32940,100)`에 넣었고
3,294,100 parameters로 FW top-1 85.04%를 얻었다. 이는 temporal position을
사용하고 JEPA mean probe보다 head가 훨씬 크므로 직접적인 representation
비교 기준으로는 부적합하다. 현재 공식 raw baseline은 temporal-mean 방식이다.

## Label-shuffled negative control

`experiment/linear_probe/label_shuffled.py`는 쉽게 제거할 수 있는 독립 control이다.
Train에서 100개 style ID에 deterministic derangement를 한 번 적용한다.

- 동일 style의 모든 train window는 동일한 가짜 label을 받는다.
- Fixed point가 없으므로 어떤 style도 자기 label로 매핑되지 않는다.
- Motion, sample ID 및 FW test label은 바꾸지 않는다.
- 실제 mapping과 shuffle seed를 summary에 저장한다.
- 정상 linear 결과와 별도 경로에 저장한다.

실행:

```bash
python -m experiment.linear_probe.label_shuffled \
  --dataset-root dataset/100style-soma77-processed \
  --device cuda:0
```

현재 결과:

| Metric | Value |
|---|---:|
| FW top-1 | 0.70% (`30/4266`) |
| FW macro accuracy | 0.58% |
| FW top-5 | 4.36% |
| FW loss | 10.3915 |

100-class balanced chance는 약 1%이고 현재 test majority baseline은 약 2.23%다.
Derangement는 fixed point가 없으므로 1%보다 조금 낮은 결과도 자연스럽다.
이 결과는 다음 leakage 가능성을 강하게 낮춘다.

- Test label이 training에 사용되는 오류
- 파일명 또는 index에서 원래 label이 classifier로 직접 유출되는 오류
- train/test를 관통하는 단순한 label-mapping leakage

따라서 random-initialized encoder의 약 50%는 평가 pipeline 오류보다는 쉬운 raw
style signal과 random-feature mapping의 결과로 해석하는 것이 타당하다.

## 현재 결과의 핵심 해석

| Baseline | FW top-1 | 해석 |
|---|---:|---|
| Balanced chance | 약 1% | 100-class 기준 |
| Label-shuffled temporal mean | 0.70% | 단순 label leakage 증거 없음 |
| Random-initialized JEPA probe | 약 50% | Random feature에도 task가 쉬움 |
| Raw temporal-mean linear | 80.08% | 평균 raw statistics에 강한 style 정보 존재 |
| Raw flattened linear, 과거 실험 | 85.04% | 시간 위치와 큰 head까지 사용한 상한 성격 |

Content-disjoint split은 동일 motion/window leakage를 방지하지만, 같은 style의
서로 다른 locomotion content가 train/test에 존재한다. Style별 자세, 속도,
보폭, 에너지 및 recording-specific nuisance가 content를 넘어 유지될 수 있다.
따라서 높은 raw accuracy 자체는 split 실패를 의미하지 않는다.

Random JEPA가 약 50%라는 사실은 반드시 baseline으로 함께 보고해야 한다.
Pretrained representation은 절대 accuracy만이 아니라 동일 probe에서
`pretrained - random initialization` 차이로 평가해야 한다.

## JEPA probe가 학습 중 하락할 때

Raw input에 정보가 있는데 JEPA probe가 학습할수록 떨어지면 style 관점에서는
우려할 만한 결과다. 다만 다음 경우를 구분해야 한다.

1. **실제 collapse 또는 정보 손실**: probe train/test가 모두 떨어지고 feature
   variance와 effective rank도 감소한다.
2. **Cross-content 일반화 실패**: probe train은 높지만 FW test만 떨어진다.
3. **Pooling bottleneck**: global mean은 떨어지지만 token flatten 또는
   mean+std probe에서는 회복한다.
4. **비선형화**: linear probe는 떨어지지만 작은 MLP/k-NN에서는 회복한다.
5. **Probe optimization mismatch**: checkpoint별 feature scale/covariance 변화로
   고정 LR head의 수렴 정도가 달라진다.
6. **Objective mismatch**: JEPA prediction에는 유용하지만 style cue는 제거한다.

다음 조건이 함께 관찰되면 JEPA representation 학습 실패라는 결론이 강해진다.

```text
learned JEPA < random JEPA
learned JEPA < parameter-comparable raw baseline
probe train accuracy도 하락
feature variance/effective rank도 악화
다른 pooling 및 downstream task에서도 회복되지 않음
```

JEPA loss가 내려가는 것만으로는 성공이라고 할 수 없다. Encoder와 target이
함께 collapse하거나 pretext shortcut을 학습해도 loss가 낮아질 수 있다.

## JEPA representation-quality monitoring

잘 학습되는지는 세 축을 함께 추적한다.

### 1. Held-out prediction quality

Pretraining dataset에서 고정된 held-out diagnostic motion을 사용한다.

- Predictor-target MSE
- Predictor-target cosine similarity
- Trivial mean predictor 대비 normalized prediction gain

권장 prediction gain:

```text
1 - MSE(predicted_target, target) / MSE(mean_target, target)
```

Training loss뿐 아니라 held-out gain이 상승해야 실제 masked target prediction이
일반화되고 있다고 볼 수 있다.

### 2. Collapse/geometry 지표

고정 sample batch에서 checkpoint마다 다음을 기록한다.

- Feature norm
- Dimension별 feature standard deviation
- Covariance effective rank
- Mean pairwise cosine similarity
- Sample 간 variance
- 한 motion 내부 token의 temporal variance

Effective rank와 std가 급격히 감소하고 pairwise cosine이 1에 접근하면 collapse
가능성이 높다.

### 3. Frozen downstream transfer

- Global-mean style linear probe의 train/FW trajectory
- Random initialization 대비 accuracy gain
- Mean+std 또는 token-level probe
- k-NN 및 cross-content style retrieval
- Root velocity/direction classification 또는 regression
- Motion phase, speed, temporal order, future-motion task
- 가능하면 각 content를 번갈아 test로 쓰는 leave-one-content-out 평가

FW trajectory는 diagnostic으로만 관찰하고 checkpoint 선택에는 사용하지 않는다.

권장 최소 dashboard:

```text
jepa/train_loss
jepa/heldout_prediction_gain
representation/feature_std
representation/effective_rank
representation/mean_pairwise_cosine
representation/temporal_variance
linear_probe/train_top1
linear_probe/test_top1
```

Random encoder, temporal-mean raw linear, chance/majority 값을 고정 기준선으로
함께 표시한다. JEPA loss 감소, held-out gain 증가, 건강한 variance/rank 유지,
여러 frozen task에서 random 대비 개선이 동시에 관찰될 때 representation이
잘 학습된다고 판단한다.

## 관련 구현 파일

- `dataset/preprocess_100style.py`: 공식 cuts 및 content-disjoint preprocessing
- `dataset/Frame_Cuts.csv`: 공식 frame cuts
- `experiment/linear_probe/train_probe.py`: offline frozen linear probe
- `experiment/linear_probe/online.py`: pretraining 중 diagnostic probe
- `experiment/linear_probe/linear.py`: temporal-mean raw single-layer baseline
- `experiment/linear_probe/label_shuffled.py`: style-level shuffled negative control
- `experiment/linear_probe/train_classifier.py`: 공용 classifier training loop
- `tests/test_preprocess_100style.py`: cut/split/preprocessing tests
- `tests/test_linear_probe.py`: offline probe tests
- `tests/test_online_probe_training.py`: online selection-isolation tests
- `tests/test_train_classifier.py`: raw linear/CNN/Transformer tests
- `tests/test_label_shuffled.py`: negative-control tests

현재 label-shuffled control을 제거하려면
`experiment/linear_probe/label_shuffled.py`와 `tests/test_label_shuffled.py`만
삭제하면 된다.
