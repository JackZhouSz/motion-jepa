# BABEL 데이터 처리 순서

MotionJEPA 사전학습에는 BONES-SEED를 사용하고, BABEL은 행동 분류 평가에 사용한다. BABEL 행동 데이터는 `preprocess_amass.py`의 출력물이 아니라, AMASS에서 변환한 SOMA77 BVH와 BABEL 주석을 직접 결합해 만든다. 따라서 이 흐름에는 `preprocess_amass.py`가 필요하지 않다.

## 입력 데이터

| 입력 | 역할 |
| --- | --- |
| AMASS `*_stageii.npz` | 원본 자세와 루트 이동, 촬영 FPS |
| BABEL `train.json`, `val.json` | `babel_sid`, AMASS 경로 `feat_p`, 행동 구간의 시작·끝 시간 및 주석 작성자 |
| BABEL `train_label_{60,120}.pkl`, `val_label_{60,120}.pkl` | 구간 ID, 클래스 번호, 시퀀스 ID, 5초 청크 번호, 주석 작성자 ID |
| `action_label_2_idx.json` | 클래스 번호와 행동 이름의 대응 관계 |

## 처리 단계

1. **AMASS를 SOMA77 BVH로 변환한다.** [`dataset/convert_amass_to_soma.py`](../dataset/convert_amass_to_soma.py)는 `*_stageii.npz`를 읽고 고정 체형의 SOMA77 골격에 맞춘다. 원본 FPS를 반올림한 값이 30의 배수인 동작만 첫 프레임부터 일정 간격으로 추출하여 30 FPS BVH를 만든다. 성공한 원본·BVH 상대 경로, FPS, 프레임 수와 추출 간격은 `conversion_manifest.jsonl`에, 제외·실패 내역은 `errors.jsonl`에 기록한다. 이 단계에서는 BABEL 주석을 사용하지 않는다.

2. **BABEL 레이블을 변환된 동작에 연결한다.** [`dataset/preprocess_babel.py`](../dataset/preprocess_babel.py)는 학습·검증 PKL의 `babel_sid`로 해당 JSON 항목을 찾는다. JSON의 `feat_p`를 AMASS `*_stageii.npz` 경로로 바꾼 뒤 변환 목록에서 BVH를 찾고, 구간 ID와 주석 작성자 ID도 대조한다. 변환되지 않았거나 BVH가 없는 동작은 건너뛰고 오류 내역에 남긴다. `frame_ann`이 있으면 이를 사용하고, 없으면 `seq_ann`의 시퀀스 전체 길이를 구간으로 사용한다.

3. **행동 구간을 5초 청크로 자른다.** 주석의 시작·끝 시간을 30 FPS 프레임 번호로 변환한다. PKL에 들어 있는 청크 번호 `chunk_n`을 사용해 `구간 시작 프레임 + chunk_n × 150`에서 시작하는 최대 150프레임을 선택한다. 150프레임은 원본 JSON의 필드가 아니라 [BABEL 행동 인식 논문](https://openaccess.thecvf.com/content/CVPR2021/papers/Punnakkal_BABEL_Bodies_Action_and_Behavior_With_English_Labels_CVPR_2021_paper.pdf)의 30 FPS, 5초 청크 설정에서 나온 값이다.

4. **MotionJEPA 표현으로 변환한다.** 청크의 SOMA77 회전을 SOMA30으로 바꾸고, 각 청크를 독립적으로 정규화 기준 자세에 맞춰 프레임당 366차원 `float32` feature를 만든다. 한 청크에 행동 레이블이 여러 개면 `.npy` 동작 파일 하나를 공유하고 레이블마다 별도의 인덱스 행을 기록한다.

5. **데이터셋과 통계를 저장한다.** `motions/{train,val}/` 아래에 `.npy` 파일을 저장하고, `train.txt`, `val.txt`, `index.json`, `meta.json`, `class-index.json`, `errors.jsonl`을 작성한다. 학습 분할의 유효 프레임으로 `stats/mean.npy`와 `stats/std.npy`를 계산한다. 공개 테스트 PKL의 클래스가 모두 `-1`이므로 이 전처리 결과의 `test.txt`는 비어 있다.

6. **학습·평가 시 로딩한다.** [`MotionDataset`](../dataset/motion_dataset.py)은 필요한 `.npy`를 불러오고, 짧은 마지막 청크를 150프레임까지 0으로 채우며 실제 길이를 함께 반환한다. 현재 구현의 이 0 패딩은 BABEL 논문에서 마지막 청크를 반복해 5초로 맞추는 방식과 다르다.

## 실행 명령

아래 명령은 `motion-jepa` conda 환경이 이미 활성화된 상태에서 저장소 루트에서 실행한다. 사전학습에는 BONES-SEED SOMA BVH와 분할 파일이 필요하다. BABEL 평가 데이터 생성에는 `dataset/amass`의 AMASS SMPL-X neutral `*_stageii.npz`, `dataset/babel-annotation`과 `dataset/babel-60-and-120`의 BABEL 파일이 필요하다. `skeleton/assets/smpl/SMPL_NEUTRAL.pkl`은 라이선스를 받은 SMPL 모델 파일이다.

### 1. BONES-SEED 사전학습 데이터 생성: 150프레임

```bash
python dataset/preprocess_bones_seed.py \
  --num_frames 150 \
  --output dataset/bones-seed-processed-nframes150
```

MotionJEPA 학습 설정의 `data.root_path`는 `./dataset/bones-seed-processed-nframes150`, `data.num_frames`는 `150`으로 맞춘다. 현재 [`configs/mjepa_1d_base.yaml`](../configs/mjepa_1d_base.yaml)은 이미 `num_frames: 150`이다.

### 2. AMASS를 SOMA77 BVH로 변환

```bash
python dataset/convert_amass_to_soma.py \
  --input-dir dataset/amass \
  --output-dir dataset/amass-soma77 \
  --smpl-model-path skeleton/assets/smpl/SMPL_NEUTRAL.pkl
```

결과는 `dataset/amass-soma77` 아래의 BVH 파일, `conversion_manifest.jsonl`, `errors.jsonl`이다. 기본값으로 이미 존재하는 BVH는 건너뛴다.

### 3. BABEL 행동 데이터 생성

```bash
python dataset/preprocess_babel.py \
  --subset 60 \
  --input-dir dataset/amass-soma77 \
  --output dataset/babel-60-processed-nframes150

python dataset/preprocess_babel.py \
  --subset 120 \
  --input-dir dataset/amass-soma77 \
  --output dataset/babel-120-processed-nframes150
```

두 명령 모두 기본 주석 경로와 레이블 경로를 사용한다. 다른 위치에 있다면 각각 `--annotations-dir`, `--labels-dir`을 지정한다.

### 4. BABEL-60/120 행동 분류기

`train_classifier.py`는 BABEL 청크 하나에 붙은 행동 레이블을 모두 정답으로 사용한다. 검증 클래스 평균 AP로 체크포인트를 선택하고 최종 검증 성능을 보고한다. 공개 test 레이블이 없으므로 결과의 `test`는 `null`이다. BABEL-120 학습 분할에 없는 클래스도 공식 클래스 번호와 출력 차원에 남긴다.

원시 동작을 입력으로 CNN과 Transformer를 각각 학습하는 명령은 다음과 같다.

```bash
python -m experiment.linear_probe.train_classifier \
  --dataset-root dataset/babel-60-processed-nframes150 \
  --input-source raw \
  --model all

python -m experiment.linear_probe.train_classifier \
  --dataset-root dataset/babel-120-processed-nframes150 \
  --input-source raw \
  --model all
```

BONES-SEED로 사전학습한 MotionJEPA 특징을 쓰려면 다음과 같이 각 데이터셋의 분류기를 실행한다. `output/YOUR_RUN/motion-jepa-1d-latest.pth.tar`는 실제 체크포인트 경로로 바꾼다. `--stats-path`에는 해당 사전학습의 정규화 통계를 지정한다.

```bash
python -m experiment.linear_probe.train_classifier \
  --dataset-root dataset/babel-60-processed-nframes150 \
  --input-source jepa \
  --jepa-checkpoint output/YOUR_RUN/motion-jepa-1d-latest.pth.tar \
  --stats-path dataset/bones-seed-processed-nframes150/stats \
  --model all

python -m experiment.linear_probe.train_classifier \
  --dataset-root dataset/babel-120-processed-nframes150 \
  --input-source jepa \
  --jepa-checkpoint output/YOUR_RUN/motion-jepa-1d-latest.pth.tar \
  --stats-path dataset/bones-seed-processed-nframes150/stats \
  --model all
```

원시 동작 결과는 `output/babel-{60,120}-classifiers/`, JEPA 특징 결과는 체크포인트 폴더의 `linear-probe/classifiers/babel-{60,120}/`에 저장된다. JEPA 특징 캐시도 `linear-probe/token-features/babel-{60,120}/`로 나뉜다.
