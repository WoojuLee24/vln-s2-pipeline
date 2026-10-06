# 261006 vln-s2-pipeline upstream 동기화 + 버전 현황 + label 품질 체크

## 1. 무엇이 달라졌나

**repo:** `/home/irteam/git/vln-s2-pipeline`
- `origin` = `WoojuLee24/vln-s2-pipeline` (fork)
- `upstream` = `KEMAL-MUDIE/vln-s2-pipeline` (원본, 이번에 추가)

**upstream 반영:** `9a8139f "061026"` (268 파일, 대부분 새 스크립트)를 fast-forward로 받았다.

**로컬 작업 커밋:** `2275855` (origin push 완료)
- `metadata_reproducer.py`: upstream head로 둔다 (v219 + `turn_threshold` 인자). 경로는 `local_paths`를 쓴다.
- `metadata_reproducer_v218.py`, `metadata_reproducer_v219.py`: 버전별로 고정해 둔 생성기
- `generate_v218_all_splits.py`: `VERSION=v218|v219`로 생성기를 고른다. LABEL 기본값은 `<VERSION>_reproduce`.

## 2. 명령어

모두 `cd /home/irteam/git/vln-s2-pipeline`에서 실행한다.

- upstream 업데이트:
  `git stash -u && git merge --ff-only upstream/master && git stash pop`
- v218 데이터 생성 (3 split):
  `python3 generate_v218_all_splits.py`
- v219 데이터 생성:
  `VERSION=v219 python3 generate_v218_all_splits.py`
  - `VERSION`: 생성기 선택 (기본 v218)
  - `--splits val_unseen`: 일부 split만 생성
  - `LABEL=<이름>`: 출력 파일 이름(`{split}_<LABEL>.json.gz`) 지정
- 결과 저장 위치: `/home/irteam/data-vol1/vln/relabel_cache/datasets/`

**검증:** 기존 `*_v218_reproduce` / `*_v219_reproduce`와 비교했다.

| VERSION | val_unseen | val_seen |
|---|---|---|
| v218 | 0/1839 diff | 0/778 diff |
| v219 | 0/1839 diff | 0/778 diff |

v218과 v219끼리는 val_unseen 116개, val_seen 72개가 다르다. train은 비교하지 않았다.

## 3. 버전 현황

스크립트 기준 마지막 버전은 **v264**이고, 버전 스크립트는 총 483개다.

| 계열 | 범위 | 방식 |
|---|---|---|
| `run_gate3_gemma_v*` | v1~v100 (97) | Gemma가 문장 생성 |
| `run_gate4_visual_v*` | v2~v203 (198) | 모든 프레임을 Gemma에 보냄 |
| `generate_v*` | v24~v238 (69) | 규칙 기반 |
| `run_gate4_*` | v240~v251 | 그 밖의 Gemma 계열 |
| `run_gt_visanchor_v*` | v252~v264 (13) | GT 문장 일부 교체 |

- README의 v272(ChronoNav, 66.2%)는 스크립트가 없고, "GT + sibling subs"라서 GT 기반이다.

**버전별로 arg/config를 줘서 실행할 수 없다.** 파일마다 상수가 하드코딩돼 있다.
- `/mnt/nvme0` 경로: 1235곳
- 이전 버전 checkpoint에 의존: 373개 (`outputs/`는 gitignore돼 있어서 이 노드에 없음)
- temperature>0: 384개 (재실행해도 같은 문장이 안 나옴)
- 계열 간 번호 충돌: 92쌍
- GT 폴더에 직접 쓰는 `DEPLOY_PATH`가 있음

**재현 가능성**
- 규칙 기반 93개: 입력만 있으면 정확히 재현할 수 있다.
- Gemma 계열 390개: Kemal의 `outputs/` checkpoint를 받아야 재현할 수 있다. 2026-10-06 요청했다.

## 4. 기존 relabel label 품질 체크 (val_unseen 209개 전부, 44초)

명령어 (파일 1개):
`python3 gate7_eval/check_label.py <val_unseen_X.json.gz> --gt <val_unseen.json.gz> --split val_unseen`
3 split 한 번에: `python3 gate7_eval/check_label.py --dir <r2r/v1 폴더> --label v218`
3 split label은 `scripts/dataset_converters/relabel_vlnce/check_label_quality.py --label all`로 한 번에 돌린다.

- 분류: vision_grounded 194, template 15
  - template: complete_metadata, gate3_gemma_v1~v6, generated_gemma, generated_gemma_visual(_v2), v4~v8
- 전부 PASS: 2개 (`gemma`, `patched`). `patched`는 GT 원본이다.
- 게이트별 FAIL 수:

| 게이트 | FAIL |
|---|---|
| G6_stop_grounded | 207 |
| G2_unique | 16 |
| G4_direction | 10 |
| G3_vocab | 1 |
| G5_distance | 1 |

- G6이 거의 전부에서 걸린다. label 문제인지 기준(25~60%)이 부적절한지 확인이 필요하다.

**검증 단계 제안 (빠른 것부터)**
1. 품질 체크: 전체 1분
2. 규칙 기반 재생성 diff: 버전당 몇 분
3. Gemma 계열 100 ep 샘플 비교
4. 공개 System2 zero-shot 평가: label당 약 1.7h (GPU). 1단계 상위 후보만 진행한다.

## 5. 다른 노드에서 v218 / v219 재실행에 필요한 것

`data-vol1`은 노드마다 디스크가 따로 있어서, 아래 데이터를 새 노드로 복사해야 한다.

**코드**
- `git clone https://github.com/WoojuLee24/vln-s2-pipeline.git` (커밋 `2275855` 이상)
- 파이썬 패키지는 추가로 설치할 게 없다. 생성 코드는 표준 라이브러리만 쓴다.
- 품질 체크: `gate7_eval/check_label.py` (+ `gate7_eval/specs/`). vln-annotator에서 복사했고 결과가 같음을 확인했다. `pyyaml`이 필요하다.

**데이터** (기본 위치: `/home/irteam/data-vol1`. 다른 위치면 환경변수로 지정한다)

| 무엇 | 경로 | 크기 | 필요? |
|---|---|---|---|
| GT | `vln/mp3d/r2r/v1/train/train.json.gz`, `val_seen/val_seen.json.gz`, `val_unseen/val_unseen_patched.json.gz` | 3 MB | 필수 |
| gate3 캐시 | `relabel_cache/gate3_perframe/{train,val_seen,val_unseen}/` | 53 MB | **필수** |
| 기존 결과 | `relabel_cache/datasets/*_v21[89]_reproduce.json.gz` | 3.4 MB | 결과가 같은지 비교할 때만 |
| 렌더 프레임 | `relabel_cache/rendered_frames/` | 2.9 GB | 불필요 (Gemma로 gate3 캐시를 다시 만들 때만) |
| gate3_landmarks | `relabel_cache/gate3_landmarks/` | 0 (비어 있음) | 불필요 |

gate3 캐시가 없어도 에러가 나지 않는다. 대신 조용히 템플릿 문장(어휘 27종)으로 생성되므로, 실행 로그의 `WARNING: ... NO gate3 perframe` 줄을 반드시 확인한다.

**경로가 다를 때**
`VLN_HABITAT_BASE=<GT v1 폴더> VLN_CACHE_ROOT=<relabel_cache 폴더> VERSION=v219 python3 generate_v218_all_splits.py`
- `VLN_HABITAT_BASE`: GT `v1` 폴더
- `VLN_CACHE_ROOT`: gate3 캐시와 결과가 있는 폴더 (`relabel_cache`)

## 6. relabel_cache 백업 (data-vol2, 공유) — 새 노드 복원

`relabel_cache`는 v218/v219 같은 규칙 기반 생성기의 **입력 캐시와 출력**을 담는 폴더다 (`local_paths.CACHE_ROOT`). `data-vol1`은 노드마다 따로라서, 공유 디스크인 `data-vol2/relabel_cache/`에 압축본을 둔다.

| 파일 | 내용 | 원본 대비 검증 |
|---|---|---|
| `gate3_perframe_20261005.tar.gz` (1.7 MB) | Gemma gate3 결과. **생성에 필수** | 13436 파일, diff 0 |
| `datasets_20261006.tar.gz` (3.4 MB) | 생성된 `*_v21[89]_reproduce` 8개 | 파일 수 8 = 8 |
| `rendered_frames_20261005.tar` (2.9 GB) | habitat 렌더 프레임. gate3를 Gemma로 다시 만들 때만 필요 | 파일 수 63901 = 63901 |

GT(`vln/mp3d/r2r/v1`)는 `data-vol2/vln/mp3d/r2r/v1`에 이미 같은 파일이 있다 (cmp 동일).

새 노드에서 복원 (필수 2개):
`mkdir -p /home/irteam/data-vol1/vln/relabel_cache && cd /home/irteam/data-vol1/vln/relabel_cache && tar xzf /home/irteam/data-vol2/relabel_cache/gate3_perframe_20261005.tar.gz && tar xzf /home/irteam/data-vol2/relabel_cache/datasets_20261006.tar.gz`

복원하지 않고 data-vol2를 바로 쓰는 경우 (GT만):
`VLN_HABITAT_BASE=/home/irteam/data-vol2/vln/mp3d/r2r/v1 VERSION=v219 python3 generate_v218_all_splits.py`
