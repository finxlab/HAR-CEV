# HAR-CEV: 레짐 전환 하에서의 구조적 확률 RV 예측

논문 *"HAR-CEV: HAR-Attentive Constant Elasticity of Variance for
Zero-Shot Realized Variance Forecasting across Economic Regimes"*
(심사 중)의 공개 코드입니다. 영어 버전은 [README.md](README.md)를
참고하세요.

HAR-CEV는 조건부 위치(location)와 스케일(scale)이 시변 CEV 확률미분방정식을
따르는 확률적 예측 모델입니다:

```
dRV = kappa_t (theta_t - RV) dt + sigma_t RV^alpha_t dW
```

* **theta_t** — *국소 HAR 앵커*: 일간/주간/월간 RV 수준의
  컨텍스트 조건부 softmax 결합.
* **kappa_t** — 상태 증폭 평균회귀 속도:
  `kappa_base,t * exp(eta_t * max(log(v_t / theta_t), 0))`.
* **sigma_t v^alpha_t** — 현재 변동성 수준에 비례해 예측구간 폭을
  조절하는 CEV 확산항.
* **ShapeNet** 헤드가 GRU 컨텍스트로부터 단조성이 보장된 7개의 표준화
  분위수 잔차를 출력합니다 — 혁신(innovation) 분포는 가우시안이 아니라
  학습됩니다.

## 저장소 구성

| 파일 | 역할 |
|---|---|
| `protocol.py` | 종목 유니버스(DJIA 30), 14개 평가 레짐, zero-shot 데이터 프로토콜 |
| `models.py` | HAR-CEV(+Exp 이산화), Anti-MR·No-SDE ablation, 손실함수 |
| `baselines.py` | HAR, DLinear, LSTM, TCN, N-BEATS, PatchTST, iTransformer (공통 분위수 헤드) |
| `compute_rv.py` | 5분봉 → 일별 연율화 실현분산 (봉 클리닝 포함) |
| `train_harcev.py` | HAR-CEV·ablation의 zero-shot 학습/추론 |
| `train_baselines.py` | 벤치마크 7종의 zero-shot 학습/추론 |
| `garch_eq.py` | GARCH(1,1) + 경험적 비율 분위수(GARCH-EQ) 베이스라인 |
| `compute_wd.py` | 쌍별 Wasserstein-1 거리 (log-RV, 학습창 vs 타깃) |
| `evaluate.py` | 논문의 모든 평가 테이블 + 패널 인식 통계 검정 |
| `var_backtest.py` | VaR-95/99 백테스트 (HS-250d·GARCH 기준, Kupiec) |

## 설치

```bash
pip install -r requirements.txt
```

Python ≥ 3.10, GPU 1장이면 충분합니다(CPU도 가능, 느릴 뿐).
(종목, 레짐, 시드) 하나의 학습은 GPU 기준 5분 이내입니다.

## 데이터 (미포함)

원본 장중 데이터는 재배포할 수 없습니다. 파이프라인은 종목별 5분봉
OHLCV 파일을 기대합니다:

```
dataset_dija/dataset_{TICKER}.csv
    index : 봉 타임스탬프 (datetime 파싱 가능)
    columns: ..., close, volume, ...
```

기간은 2000-01-01부터 마지막 평가 레짐까지입니다. 정규장 5분 종가만
있으면 어떤 데이터 벤더든 무방합니다. `compute_rv.py`가
`dataset_rv/rv_{TICKER}.csv`(일별 연율화 RV)를 생성하며, 모델은 이
파일만 사용합니다.

## 논문 재현 순서

```bash
# 1. 5분봉 → 일별 연율화 RV
python compute_rv.py --data_dir dataset_dija --out_dir dataset_rv

# 2. HAR-CEV + ablation (zero-shot 411쌍, 시드 3개)
python train_harcev.py --rv_dir dataset_rv

# 3. 벤치마크 7종
python train_baselines.py --rv_dir dataset_rv

# 4. GARCH(1,1) 경험적 비율 분위수 베이스라인
python garch_eq.py --data_dir dataset_dija --rv_dir dataset_rv

# 5. 쌍별 분포 이동 측정
python compute_wd.py --rv_dir dataset_rv

# 6. 전체 평가 테이블 + 패널 인식 검정
python evaluate.py --pred_dir outputs

# 7. VaR 백테스트
python var_backtest.py --data_dir dataset_dija --rv_dir dataset_rv
```

2–3단계가 가장 오래 걸리며(411쌍 × 모델 × 시드 3), 체크포인트 디렉터리를
통해 이어서 실행할 수 있고 `--ticker` / `--model(s)` 필터로 부분 실행이
가능합니다.

## 프로토콜 요약

* **Zero-shot 쌍**: 30개 종목 × 14개 레짐 각각에 대해, 직전 756 거래일
  (~3년)로 학습한 뒤 타깃 레짐의 모든 날을 **어떤 적응도 없이** 예측.
  756일 미만 쌍은 제외되어 411쌍이 유효합니다.
* **타깃**: 다음날 연율화 실현분산의 7개 분위수(5–95%). 모든 모델은
  동일한 최적화 스케줄(Adam 1e-3, ReduceLROnPlateau, 조기종료,
  최대 150 에폭, 배치 1024, 시드 3개 앙상블)에서 평균 pinball 손실로
  학습됩니다.
* **입력 클리핑**: 모델 입력은 학습창 백분위수로 인과적으로 클리핑
  (벤치마크: p99 상단 클립, HAR-CEV: 불필요 —
  `--clip_lower/--clip_upper` 참조). 타깃은 클리핑하지 않습니다.
* **평가지표**: ND, MAE, WIS(weighted interval score; CRPS의 분위수
  근사), TailAE(중앙값 절대오차의 CVaR-95), NIS90(정규화 구간 점수),
  PS-WIS/PS-Bias(스파이크 직후 구간), CalibErr-7, TailCalib.
* **분포 이동**: 실제 756일 학습창과 타깃 레짐 간 log-RV 공간
  Wasserstein-1 거리. 411쌍을 4분위(Q1–Q4)로 층화합니다.

## 인용

```bibtex
@inproceedings{harcev2026,
  title     = {HAR-CEV: HAR-Attentive Constant Elasticity of Variance for
               Zero-Shot Realized Variance Forecasting across Economic
               Regimes},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
