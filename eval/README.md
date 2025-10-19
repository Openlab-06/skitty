# 🏆 Spam XAI Evaluation System

스팸 탐지 설명 가능성 AI (Explainable AI) 평가 시스템입니다.

## 📊 평가 메트릭

세 가지 메트릭을 조합하여 스팸 탐지 설명의 품질을 종합 평가합니다:

### 1️⃣ BLEU Score (가중치: 25%)

**목적**: 어휘 수준의 일치도 측정

- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 생성된 설명이 기준 설명과 얼마나 많은 단어를 공유하는지 측정
- **특징**: 
  - n-gram 기반의 정확한 일치도 계산
  - 같은 의미라도 다른 단어 사용 시 점수 감소

### 2️⃣ Semantic Similarity (가중치: 35%)

**목적**: 의미론적 유사성 측정

- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **설명**: 생성된 설명의 의미가 기준 설명과 얼마나 일치하는지 측정
- **기술**: Cross-Encoder (qnli-distilroberta-base) 기반
- **특징**:
  - 문맥을 고려한 의미 비교
  - 다른 단어지만 같은 의미면 높은 점수
  - 설명의 핵심 내용이 일치하는지 확인

### 3️⃣ LLM Judge Score (가중치: 40%)

**목적**: 설명 품질의 종합 평가

- **범위**: 0.0 ~ 1.0 (높을수록 좋음)
- **판사 모델**: GPT-4o (SOTA 모델)
- **평가 기준**:
  - **정확성 (Accuracy)**: 설명이 스팸 판정의 실제 이유와 일치하는가?
  - **완전성 (Completeness)**: 모든 스팸 판정 근거를 포함했는가?
  - **명확성 (Clarity)**: 설명이 명확하고 이해하기 쉬운가?
  - **구체성 (Specificity)**: 구체적인 근거를 제시했는가?
- **특징**:
  - 인간 수준의 이해도로 평가
  - XAI 설명의 실질적 품질 판단

## 🎯 종합 점수 (Overall Score)

```
Overall Score = BLEU × 0.25 + Semantic × 0.35 + LLM Judge × 0.40
```

**가중치 근거**:
- **BLEU (25%)**: 기본적인 일치도는 필요하지만 완벽한 일치는 아님
- **Semantic (35%)**: 의미가 가장 중요하므로 가장 높은 비중
- **LLM Judge (40%)**: 최종 품질 판단으로 가장 높은 비중

## 🚀 사용 방법

### 1. 기본 사용

```bash
./scripts/run_evaluate.sh
```

전체 데이터셋으로 평가를 실행합니다.

### 2. 샘플 수 제한 (빠른 테스트)

```bash
./scripts/run_evaluate.sh --limit 10
```

처음 10개 샘플만으로 평가합니다.

### 3. 상세 로그 출력

```bash
./scripts/run_evaluate.sh --limit 50 --verbose
```

평가 진행 상황을 상세히 출력합니다.

### 4. 특정 모델 평가

```bash
./scripts/run_evaluate.sh --model "google/gemma-3-4b-it"
```

지정한 모델로 평가합니다.

### 5. 복합 옵션

```bash
./scripts/run_evaluate.sh --limit 100 --verbose --model "google/gemma-3-4b-it"
```

100개 샘플을 상세 로그와 함께 평가합니다.

## 📋 Python 직접 실행

```bash
# 기본 실행 (전체 데이터셋)
python eval/evaluation.py

# 10개 샘플 테스트
python eval/evaluation.py --limit 10

# 상세 로그 출력
python eval/evaluation.py --limit 50 --verbose true

# 특정 모델로 평가
python eval/evaluation.py --model "google/gemma-3-4b-it"

# 모든 옵션 적용
python eval/evaluation.py --limit 100 --verbose true --model "google/gemma-3-4b-it"
```

## 📊 리더보드 출력 예시

```
================================================================================
                 🏆 SPAM XAI EVALUATION LEADERBOARD 🏆
================================================================================

Model                BLEU         Semantic     LLM Judge    Overall
--------------------------------------------------------------------
google/gemma-3-4b-it 0.7234       0.8456       0.8102       0.8173

--------------------------------------------------------------------------------
                           📊 Metric Details
--------------------------------------------------------------------------------

✓ BLEU Score (어휘 일치도): 0.7234
  - 범위: 0~1 (높을수록 좋음)
  - 설명이 기준 설명과 얼마나 일치하는지 측정

✓ Semantic Similarity (의미론적 유사성): 0.8456
  - 범위: 0~1 (높을수록 좋음)
  - 생성된 설명의 의미가 기준과 일치하는 정도

✓ LLM Judge Score (설명 품질): 0.8102
  - 범위: 0~1 (높을수록 좋음)
  - 정확성, 완전성, 명확성, 구체성을 GPT-4o가 평가

✓ Overall Score (종합 점수): 0.8173
  - BLEU 25% + Semantic 35% + LLM Judge 40%

================================================================================
평가된 샘플 수: 100
================================================================================
```

## 💾 결과 저장

평가 완료 후 다음 파일이 생성됩니다:

- **`./eval/results.json`**: 상세 평가 결과
  - 종합 점수 (summary)
  - 각 샘플별 상세 점수 (details)
  - LLM Judge 설명

### results.json 구조

```json
{
  "summary": {
    "bleu": 0.7234,
    "semantic_similarity": 0.8456,
    "llm_judge": 0.8102,
    "total_samples": 100
  },
  "details": [
    {
      "sample_id": 0,
      "reference": "스팸 판정 이유: ...",
      "prediction": "생성된 설명: ...",
      "bleu_score": 0.75,
      "semantic_similarity": 0.82,
      "llm_judge_score": 0.88,
      "llm_judge_explanation": "정확하고 구체적인 설명..."
    },
    ...
  ]
}
```

## 🔧 모듈 구조

```
eval/
├── __init__.py
├── evaluation.py          # 메인 평가 클래스
└── README.md             # 이 파일
```

### EvaluationMetrics 클래스

샘플별 평가 메트릭을 저장하는 데이터클래스:

```python
@dataclass
class EvaluationMetrics:
    sample_id: int
    prompt: str
    reference: str
    prediction: str
    bleu_score: float
    semantic_similarity: float
    llm_judge_score: float
    llm_judge_explanation: str
```

### LLMEvaluation 클래스

주요 메서드:

| 메서드 | 설명 |
|--------|------|
| `__init__()` | 평가기 초기화 |
| `evaluate()` | 전체 평가 실행 |
| `_calculate_semantic_similarity()` | 의미론적 유사성 계산 |
| `_llm_judge_evaluation()` | LLM Judge 평가 |
| `print_leaderboard()` | 리더보드 출력 |
| `save_detailed_results()` | 결과 저장 |

## 📦 의존성

- `opik`: BLEU 메트릭 계산
- `sentence-transformers`: Semantic Similarity 계산
- `openai`: LLM Judge (GPT-4o) 호출
- `datasets`: Spam_QA-Corpus 로드

## ⚙️ 환경 설정

다음 환경 변수가 필요합니다:

```bash
# 평가 대상 모델
SPAM_MODEL_URL="http://localhost:8000/v1"
SPAM_MODEL_API_KEY="your-api-key"
SPAM_MODEL="google/gemma-3-4b-it"

# LLM Judge (GPT-4o)
OPENAI_API_KEY="sk-..."
```

## 🎓 평가 해석 가이드

| 범위 | 해석 | 평가 |
|------|------|------|
| 0.90 ~ 1.00 | 매우 우수 | ⭐⭐⭐⭐⭐ |
| 0.80 ~ 0.89 | 우수 | ⭐⭐⭐⭐ |
| 0.70 ~ 0.79 | 양호 | ⭐⭐⭐ |
| 0.60 ~ 0.69 | 보통 | ⭐⭐ |
| 0.00 ~ 0.59 | 개선 필요 | ⭐ |

## 🔄 모델 비교

여러 모델을 순차적으로 평가하여 비교할 수 있습니다:

```bash
# Model 1
python eval/evaluation.py --limit 100 --model "model1"

# Model 2
python eval/evaluation.py --limit 100 --model "model2"

# results.json 비교
```

## 🐛 문제 해결

### CrossEncoder 모델 다운로드 오류

```bash
python -m sentence_transformers.models download cross-encoder/qnli-distilroberta-base
```

### OpenAI API 오류

```bash
# API 키 확인
echo $OPENAI_API_KEY

# 연결 테스트
python -c "from openai import OpenAI; print(OpenAI(api_key='sk-...').models.list())"
```

## 📚 참고 자료

- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [Semantic Similarity with Transformers](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [LLM as Judge](https://arxiv.org/abs/2306.05685)

## 🤝 기여

개선 사항 및 버그 리포트는 이슈로 등록해주세요!
