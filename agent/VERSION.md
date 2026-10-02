# BONES-SEED 전처리와 마스킹 버전

| 버전 | 전처리 | 마스킹 |
| --- | --- | --- |
| v1 | `dataset/preprocess_bones_seed.py`: 완전한 고정 길이 window만 저장한다. 기본 150프레임보다 짧은 원본과 마지막 window 이후의 잔여 구간은 제외한다. | 기존 1D/2D collator. 배치의 최소 유효 길이로 block 길이를 정하고, context도 배치 최소 개수로 줄인다. 1D의 `context_selection`은 `prefix`/`random`을 지원한다. |
| v2 | `dataset/preprocess_bones_seed_v2.py`: 기본 최소 **60프레임(30 FPS에서 2초)**. 60–149프레임 원본은 실제 길이로 저장하고 loader가 padding한다. 긴 원본은 끝에 맞춘 마지막 완전한 window를 추가해 잔여 구간을 보존한다. | `mask/collators_v2.py`: **1D frame/patch만 지원**한다. sample별 유효 길이에 비례해 block을 만들고 context는 `all`/`prefix`/`random`으로 선택한다. 개수가 다른 index는 `-1`로 padding하며 encoder/predictor attention과 loss에서 제외한다. 2D v2는 아직 구현하지 않았다. |

## v2 설정

- 기본 길이: `--num_frames 150 --min_frames 60 --fps 30`.
- `mask.version: v2`로 선택한다. 생략하면 기존 v1을 사용한다.
- `min_context_ratio: 0.2`, `min_context_tokens: 1`: sample마다 유효 token의 최소 20%를 context로 남긴다. target끼리는 겹칠 수 있으며, `allow_overlap: false`이면 context와 target은 겹치지 않는다.
- `context_selection: all`은 context를 전부 보존하는 기본 동작이다. `prefix`는 앞에서부터, `random`은 전체 유효 context 후보를 `randperm`한 뒤 필요한 개수를 선택하고 시간순으로 정렬한다.
- `prefix`/`random`은 배치 전체 context의 최소 개수를 기준으로 줄인다. 각 sample의 최소 context 보장보다 적게 줄이지 않으며, 이 때문에 남는 길이 차이는 padding으로 맞춘다. Target block은 sample별 길이를 그대로 유지한다.
- 두 v2 base config는 `context_selection: random`을 사용한다.
- Patch에서는 완전한 patch만 유효하다. 예를 들어 patch 3, 실제 길이 61이면 20개 token을 사용한다.
- Loss는 유효 target token만 계산하고 각 sample/context/target 조합에 동일한 가중치를 준다.
- 전처리 정책과 최소 길이는 metadata에 기록한다. 다른 정책의 출력은 자동 재사용하지 않는다. v1/v2 mask checkpoint state는 서로 이어서 학습할 수 없다.
- Context 선택 방식도 checkpoint에 저장한다. 선택 방식이 기록되지 않은 이전 v2 checkpoint는 `all`로 해석하며, 다른 방식으로 정확히 재개하는 것은 허용하지 않는다.

## 실행

저장소 루트에서 실행한다.

```bash
python -m dataset.preprocess_bones_seed_v2 \
  --num_frames 150 --min_frames 60 --fps 30 \
  --output dataset/bones-seed-processed-v2-nframes150-min60

# Frame token 1D base
python main.py --config configs/mjepa_1d_base_v2.yaml

# Temporal patch 3, 1D base
python main.py --config configs/mjepa_patch_1d_base_v2.yaml
```

두 config는 새 데이터와 별도의 출력 폴더를 사용한다. Segmentation probe는 유지하고 classification linear/attentive probe는 비활성화한다.
