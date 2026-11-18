# GAZERA

**Intelligent Eye-Tracking Data Analytics** · *From Gaze Data to Visual Intelligence.*
Import · Analyze · Visualize · Interpret

Clean-room, vendor-neutral, **MIT-licensed** research software by
**FEI · HEL Lab, Gyeongsang National University**.

- Web workbench (browser only, no install): <https://sunbear5020.github.io/gazera/>
- Python engine: this repository / this folder (`deploy/gazera/`)
- Version: **0.1.0 (GAZERA Core)**

> GAZERA is an independent open-source project developed by FEI · HEL Lab
> (Gyeongsang National University) under the MIT License. It is not affiliated
> with, endorsed by, or sponsored by Tobii AB. All product names and trademarks
> belong to their respective owners.

---

## 한국어

### 무엇인가

GAZERA는 **시선추적 장비에서 내보낸 표준 구분자 텍스트(CSV/TSV)를 읽어** 데이터
품질을 점검하고, 응시(Fixation)·도약(Saccade)·관심영역(AOI) 지표를 산출하고,
조건 간 통계 비교와 Excel 보고서까지 만들어 주는 연구용 소프트웨어입니다.
측정·장비 제어 프로그램이 아니라, **이미 측정된 자료를 연구 목적에 맞게 다시
분석하는 엔진**입니다.

### 클린룸 · 라이선스 원칙 (최우선)

| 원칙 | 내용 |
| --- | --- |
| 라이선스 | MIT · `Copyright (c) 2026 FEI · HEL Lab, Gyeongsang National University` |
| 벤더 중립 | CSV/TSV만 읽습니다. SDK·DLL·전용 바이너리 포맷을 쓰지 않습니다. |
| 금지 항목 | 벤더 SDK/바이너리, 공식 샘플 파일 재배포, 로고·스크린샷·UI 카피, 비공개 포맷 역공학 코드 — **일체 없음** |
| 알고리즘 | I-VT(속도 임계값) 분류는 공개 문헌(Salvucci & Goldberg, 2000)을 근거로 자체 구현 |
| 데이터 | 데모·테스트 데이터는 100 % 자체 생성 합성 데이터. 실제 피험자 자료 없음 |
| 상표 | 제품명·상표는 각 소유자에게 있으며, GAZERA는 Tobii AB와 제휴·후원·보증 관계가 없습니다 |

컬럼명(`Recording timestamp`, `Gaze point X`, `Eye movement type` 등)은 **자동
매핑을 위한 패턴**으로만 사용합니다. 어떤 이름 구조든 수동 매핑으로 바꿀 수
있습니다.

### 설치

```bash
cd deploy/gazera
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3분 안에 실행

```bash
# 1) 합성 데모 데이터 생성 (도시숲 vs 도심 가로)
python -m gazera_core demo --out demo.tsv

# 2) 분석 → Excel(.xlsx 다중 시트) · JSON · CSV
python -m gazera_core analyze demo.tsv --out results

# 3) 로컬 API
uvicorn api.server:app --reload --port 8000
#    http://127.0.0.1:8000/docs 에서 /api/analyze, /api/export 확인
```

웹 워크벤치는 설치 없이 <https://sunbear5020.github.io/gazera/> 에서 바로
쓸 수 있으며, 파일은 브라우저 안에서만 처리됩니다(외부 전송 없음).

### 파이프라인

```
CSV/TSV ──▶ loader ──▶ cleaner ──▶ ivt ──▶ metrics ──▶ stats ──▶ exporter
           인코딩·구분자   품질검사     I-VT     Fix/Sac/AOI   조건 비교   xlsx/json/csv
           헤더·컬럼 매핑   결측·유효율   분류      지표 산출
```

| 모듈 | 역할 |
| --- | --- |
| `gazera_core/loader.py` | 인코딩(UTF-8/CP949/EUC-KR/UTF-16) · 구분자(`, \t ; \|`) · 머리말 자동 감지, 벤더 중립 컬럼 자동 매핑 + 수동 매핑 |
| `gazera_core/cleaner.py` | 행수·유효 시선 비율·추정 Hz·결측률·타임스탬프 단조성·화면 밖 좌표 품질 보고, 짧은 결측 보간 |
| `gazera_core/ivt.py` | 원시 좌표만 있을 때 I-VT로 Fixation/Saccade 자체 분류 (화면 기하 기반 px/deg 추정) |
| `gazera_core/metrics.py` | 응시 횟수·총/평균/중앙 시간, 도약 빈도·진폭·속도, AOI 체류시간·방문횟수·TTFF |
| `gazera_core/stats.py` | 기술통계, 정규성(Shapiro-Wilk) 후 t-test/Mann-Whitney/Wilcoxon/ANOVA/Kruskal-Wallis, 효과크기, Holm 보정 |
| `gazera_core/exporter.py` | Excel 다중 시트, CSV(UTF-8 BOM), JSON |
| `gazera_core/synthetic.py` | 100 % 합성 시선 데이터 생성기 (도시숲 vs 도심 가로, 정답 라벨 포함) |
| `gazera_core/pipeline.py` | 위 전 과정을 하나로 묶는 `analyze()` / `analyze_file()` |
| `api/server.py` | `/api/health` · `/api/sample` · `/api/analyze` · `/api/export` |

### 컬럼 자동 매핑

자동 인식되는 표준 필드(대소문자·공백 무시):

- **시각**: `Recording timestamp`, `Timestamp`, `Device timestamp`, …
- **시선 좌표**: `Gaze point X/Y`, `Gaze point X (ADCS px)`, `Gaze x/y`, …
- **동공**: `Pupil diameter left/right`, `Pupil left/right`, `Pupil diameter`, …
- **유효성**: `Validity left/right`, `Validity`, `Gaze validity`
- **이벤트**: `Eye movement type`, `Event type`, `Gaze event duration`
- **응시점**: `Fixation point X/Y`
- **피험자/조건**: `Participant name`, `Recording name`, `Presented stimulus name`, `Condition`
- **AOI**: `AOI hit [영역이름]` (여러 개 자동 인식)

수동 매핑:

```python
import gazera_core as gz

data = gz.load_gaze(
    "my_export.tsv",
    mapping={"gaze_x": "X좌표", "gaze_y": "Y좌표", "timestamp": "시간(ms)",
             "subject": "피험자ID", "condition": "자극"},
)
result = gz.analyze(data)          # AnalysisConfig() 로 임계값·AOI 조정
result.sheets()                     # DataFrame 묶음
result.to_dict()                    # JSON 직렬화 가능한 보고서
```

### 분석 파라미터

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `velocity_threshold_deg_s` | 30 | 이 값보다 느리면 응시, 빠르면 도약 |
| `min_fixation_ms` | 60 | 이보다 짧은 응시는 주변에 흡수 |
| `min_saccade_ms` | 20 | 이보다 짧은 도약은 주변에 흡수 |
| `max_sample_gap_ms` | 75 | 이보다 긴 공백은 별개 구간으로 분리 |
| `smoothing_samples` | 1 | 속도 계산 전 이동평균 창 |
| `viewing_distance_mm` / `screen_width_mm` | 650 / 530 | px→deg 환산 (또는 `px_per_deg` 직접 지정) |

AOI는 정규화(0–1) 사각형 또는 픽셀 사각형으로 지정합니다. 기본값은 화면을
가로로 3등분한 `상단 · 중앙 · 하단`입니다.

```python
from gazera_core import AOI, AnalysisConfig, IVTConfig, analyze

cfg = AnalysisConfig(
    ivt=IVTConfig(velocity_threshold_deg_s=35, min_fixation_ms=80),
    aois=[AOI("수관", 0.0, 0.0, 1.0, 0.4), AOI("숲길", 0.2, 0.6, 0.6, 0.4)],
)
result = analyze(data, cfg)
```

### 결과물

- `results/<파일명>_gazera.xlsx` — README, Summary, Quality, Quality by recording,
  Participants, Conditions, Statistics, AOI, Fixation by condition, Saccade by condition
- `results/<파일명>_gazera.json` — 전체 보고서 (웹/API와 동일 스키마 `gazera.result.v1`)
- `results/*.csv` — 시트별 CSV

### 테스트

```bash
python -m pytest            # 65 tests
python -m pytest -k ivt -v
```

테스트는 모두 합성 데이터로 돌며, 알려진 정답(생성기가 만든 응시 개수)과
비교해 I-VT 분류를 검증합니다.

### 3단계 로드맵

| 단계 | 이름 | 내용 | 상태 |
| --- | --- | --- | --- |
| 1차 | **GAZERA Core** | 파일 자동 인식, 품질검사, Fixation·Saccade·AOI, 기술통계, Excel 출력 | **이 버전 (0.1.0)** |
| 2차 | **GAZERA Visual** | Heatmap·Scanpath·응시순서·AOI 전이행렬·피험자/집단 비교 (Python: PNG/PDF) | 웹 워크벤치에 미리보기 구현 |
| 3차 | **GAZERA AI** | 환경영상 자동 분할, 시선–환경요소 매칭, 시선패턴 군집, HRV 등 생리자료 통합 | 설계 단계 |

> 웹 워크벤치는 1차 기능에 더해 Heatmap/Scanpath **미리보기**를 브라우저
> Canvas로 제공합니다(2차 엔진의 정식 출력은 PNG/PDF 내보내기로 확장 예정).

---

## English

### What it is

GAZERA reads **ordinary CSV/TSV exports from any eye-tracking device**,
inspects data quality, derives fixation / saccade / AOI metrics with its own
velocity-threshold implementation, compares conditions statistically, and writes
a multi-sheet Excel report. It is an **analysis engine for data you already
recorded**, not a recording or device-control application.

### Clean-room and licence principles

- **MIT License** — `Copyright (c) 2026 FEI · HEL Lab, Gyeongsang National University`.
- Only delimited text is read; no SDK, DLL, installer or proprietary binary format.
- No vendor sample files, logos, screenshots, fonts or UI copies are redistributed.
- No reverse-engineered format code. The I-VT classifier follows the published
  literature (Salvucci & Goldberg, 2000), not any vendor implementation.
- All demo and test data is 100 % synthetic and generated in-repo.
- Third-party libraries are used as dependencies only; see `NOTICE.md`.

### Install and run

```bash
pip install -r requirements.txt
python -m gazera_core demo --out demo.tsv
python -m gazera_core analyze demo.tsv --out results
uvicorn api.server:app --port 8000      # http://127.0.0.1:8000/docs
python -m pytest
```

Or open the browser workbench: <https://sunbear5020.github.io/gazera/> — the
file never leaves the browser.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | version, licence and non-affiliation notice |
| GET | `/api/sample?subjects=6&duration=6` | synthetic demo dataset (TSV) |
| POST | `/api/analyze` | multipart `file` + optional `mapping` / `config` JSON |
| POST | `/api/export?fmt=xlsx\|csv\|json` | same input, returns a file |

```bash
curl -F "file=@demo.tsv" http://127.0.0.1:8000/api/analyze | jq .summary
curl -F "file=@demo.tsv" -o report.xlsx http://127.0.0.1:8000/api/export
```

### Metric definitions

- **Fixation count** — number of I-VT fixation segments.
- **Fixation duration** — segment duration (last minus first sample timestamp);
  total, mean, median, min, max reported.
- **Saccade amplitude** — angular distance between the centroids of the first
  and last three samples of a saccade, in visual degrees.
- **Saccade velocity** — mean and peak deg/s inside the segment.
- **AOI dwell** — summed fixation duration whose centroid falls inside the AOI.
- **AOI dwell ratio** — dwell / total fixation duration of the recording.
- **AOI visits** — number of fixations entering the AOI.
- **TTFF** — time from recording start to the first fixation inside the AOI.

Statistics: Shapiro-Wilk per group; Welch t-test or Mann-Whitney U for two
groups; paired t / Wilcoxon signed-rank for repeated measures; one-way ANOVA or
Kruskal-Wallis for >2 groups; Cohen's d / rank-biserial effect sizes; optional
Holm correction across dependent variables.

### Repository layout

```
gazera/
├── LICENSE
├── NOTICE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── api/
│   └── server.py
├── gazera_core/
│   ├── __init__.py  __main__.py  cli.py
│   ├── loader.py  cleaner.py  ivt.py
│   ├── metrics.py  stats.py  exporter.py
│   ├── synthetic.py  pipeline.py
├── examples/
│   └── (generated demo data lands here in the docs example)
└── tests/
    ├── conftest.py
    ├── test_api.py  test_cleaner.py  test_ivt.py
    ├── test_loader.py  test_metrics.py  test_stats.py
    └── test_synthetic_pipeline.py
```

### Citation

> FEI · HEL Lab (2026). GAZERA: Intelligent Eye-Tracking Data Analytics
> (Version 0.1.0) [Computer software]. Gyeongsang National University. MIT License.

### Contact

FEI · HEL Lab — Department of Forest Environmental Resources,
Gyeongsang National University · `bkim5020@gnu.ac.kr`
