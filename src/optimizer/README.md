# 🎯 모델 양자화 가이드

학습된 LLM 모델을 AWQ 또는 GPTQ로 양자화하여 추론 속도를 높이고 메모리 사용량을 줄이는 도구입니다.

## 📋 목차

- [개요](#개요)
- [양자화 타입 비교](#양자화-타입-비교)
- [설치](#설치)
- [사용법](#사용법)
- [고급 설정](#고급-설정)
- [문제 해결](#문제-해결)

## 🎯 개요

### 양자화란?

양자화(Quantization)는 모델의 가중치와 활성화 값을 낮은 정밀도(예: 4bit)로 변환하여 모델 크기를 줄이고 추론 속도를 높이는 기술입니다.

**장점:**
- 🚀 **추론 속도 향상**: 2-4배 빠른 추론
- 💾 **메모리 사용량 감소**: 모델 크기 75% 감소 (16bit → 4bit)
- 💰 **비용 절감**: 더 작은 GPU로 서비스 가능
- 📊 **정확도 유지**: 최소한의 성능 손실 (1-3%)

## ⚖️ 양자화 타입 비교

### AWQ (Activation-aware Weight Quantization)

**특징:**
- ✅ **빠른 추론 속도**: GPTQ보다 1.5-2배 빠름
- ✅ **높은 정확도**: Activation을 고려한 가중치 양자화
- ✅ **간단한 설정**: 자동으로 최적화
- ⚠️ **제한적 지원**: 일부 하드웨어에서만 작동

**추천 상황:**
- 최신 CUDA GPU (Ampere 이상)
- 최대 성능이 필요한 경우
- Gemma, Llama 등 주요 모델

```bash
# AWQ 양자화 (권장)
./scripts/run_quantize.sh --quantization-type awq
```

### GPTQ (Generative Pre-trained Transformer Quantization)

**특징:**
- ✅ **안정적**: 광범위한 테스트와 검증
- ✅ **범용성**: 대부분의 GPU에서 작동
- ✅ **커뮤니티 지원**: 많은 문서와 예제
- ⚠️ **느린 속도**: AWQ보다 느림

**추천 상황:**
- 구형 GPU (Pascal, Turing)
- AWQ가 작동하지 않을 때
- 안정성이 중요한 프로덕션 환경

```bash
# GPTQ 양자화
./scripts/run_quantize.sh --quantization-type gptq
```

### 비교 표

| 항목 | AWQ | GPTQ |
|------|-----|------|
| 추론 속도 | ⚡⚡⚡ 매우 빠름 | ⚡⚡ 빠름 |
| 정확도 | 🎯 우수 | 🎯 양호 |
| 메모리 사용 | 💾 낮음 | 💾 낮음 |
| GPU 호환성 | 🔧 제한적 | 🔧 범용 |
| 설정 난이도 | 😊 쉬움 | 😊 쉬움 |
| 양자화 시간 | ⏱️ 빠름 | ⏱️ 중간 |

## 🔧 설치

### 1. AWQ 설치

```bash
# AWQ 라이브러리 설치
pip install autoawq

# 또는 최신 버전 (GPU 지원)
pip install autoawq --extra-index-url https://wheels.autoawq.ai/
```

### 2. GPTQ 설치

```bash
# GPTQ 라이브러리 설치
pip install auto-gptq

# 또는 CUDA 버전 지정 (예: CUDA 11.8)
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

### 3. 의존성 확인

```bash
# PyTorch 및 transformers 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 🚀 사용법

### 기본 워크플로우

```bash
# 1단계: 모델 학습
./scripts/run_train.sh

# 2단계: LoRA 어댑터 병합
./scripts/run_merge.sh

# 3단계: 모델 양자화
./scripts/run_quantize.sh
```

### 기본 양자화 (AWQ)

```bash
# 기본 설정으로 AWQ 양자화 (4bit, 128 group size)
./scripts/run_quantize.sh

# 또는 직접 Python 스크립트 실행
python -m src.optimizer.quantize \
  --model-path ./outputs/merged \
  --output-dir ./outputs/quantized/awq \
  --quantization-type awq
```

**결과:**
- 입력: `./outputs/merged` (병합된 모델)
- 출력: `./outputs/quantized/awq` (4bit AWQ 양자화 모델)

### GPTQ 양자화

```bash
# GPTQ로 양자화
./scripts/run_quantize.sh --quantization-type gptq

# 출력 경로 지정
./scripts/run_quantize.sh \
  --quantization-type gptq \
  --output-dir ./outputs/quantized_models
```

### 8bit 양자화

```bash
# 더 높은 정확도가 필요한 경우 8bit 양자화
./scripts/run_quantize.sh --bits 8

# 또는 GPTQ 8bit
./scripts/run_quantize.sh --quantization-type gptq --bits 8
```

**비트 수별 특징:**
- **4bit**: 최대 압축, 빠른 속도, 약간의 정확도 손실
- **8bit**: 균형잡힌 성능, 최소한의 정확도 손실

### Calibration 샘플 수 조정

```bash
# 더 많은 샘플로 정확도 향상 (시간 증가)
./scripts/run_quantize.sh --calibration-samples 1024

# 빠른 테스트용 (적은 샘플)
./scripts/run_quantize.sh --calibration-samples 128
```

**권장 샘플 수:**
- **빠른 테스트**: 128 샘플
- **일반 사용**: 512 샘플 (기본값)
- **높은 정확도**: 1024-2048 샘플

## 🎛️ 고급 설정

### 전체 옵션

```bash
./scripts/run_quantize.sh \
  --model-path ./outputs/merged \
  --output-dir ./outputs/quantized \
  --quantization-type awq \
  --bits 4 \
  --group-size 128 \
  --calibration-samples 512 \
  --dataset-name Devocean-06/Spam_QA-Corpus \
  --dataset-split train \
  --max-seq-length 1500
```

### 파라미터 설명

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `--model-path` | 양자화할 모델 경로 | `./outputs/merged` | - |
| `--output-dir` | 저장 경로 | `./outputs/quantized` | - |
| `--quantization-type` | 양자화 타입 | `awq` | `awq` (빠름), `gptq` (안정) |
| `--bits` | 양자화 비트 | `4` | `4` (압축), `8` (정확도) |
| `--group-size` | 그룹 사이즈 | `128` | `128` (일반), `64` (정확도) |
| `--calibration-samples` | Calibration 샘플 수 | `512` | `512-1024` |
| `--max-seq-length` | 최대 시퀀스 길이 | `1500` | 모델 설정과 동일 |

### 다양한 설정 예시

#### 1. 빠른 프로토타이핑

```bash
# 적은 샘플, 빠른 양자화
./scripts/run_quantize.sh \
  --calibration-samples 128 \
  --bits 4
```

#### 2. 프로덕션 배포

```bash
# 높은 정확도, 최적화된 설정
./scripts/run_quantize.sh \
  --quantization-type awq \
  --bits 4 \
  --group-size 128 \
  --calibration-samples 1024
```

#### 3. 메모리 제약 환경

```bash
# 최대 압축
./scripts/run_quantize.sh \
  --bits 4 \
  --calibration-samples 256
```

#### 4. 정확도 우선

```bash
# 8bit 양자화, 많은 calibration 샘플
./scripts/run_quantize.sh \
  --bits 8 \
  --group-size 64 \
  --calibration-samples 2048
```

## 📊 성능 벤치마크

### Gemma-3-4B 모델 기준

| 설정 | 모델 크기 | 추론 속도 | 정확도 | 메모리 |
|------|----------|----------|--------|--------|
| FP16 (원본) | ~8GB | 1x | 100% | ~10GB |
| AWQ 4bit | ~2GB | 3.5x | 98% | ~3GB |
| GPTQ 4bit | ~2GB | 2.5x | 97% | ~3GB |
| AWQ 8bit | ~4GB | 2x | 99% | ~5GB |

**측정 환경**: NVIDIA A100, Batch Size 1, Seq Length 512

## 🔍 양자화된 모델 사용

### 1. Python에서 로드

#### AWQ 모델

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 양자화된 모델 로드
model = AutoAWQForCausalLM.from_quantized(
    "./outputs/quantized/awq",
    fuse_layers=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./outputs/quantized/awq")

# 추론
inputs = tokenizer("스팸 문자를 판정해주세요", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

#### GPTQ 모델

```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# 양자화된 모델 로드
model = AutoGPTQForCausalLM.from_quantized(
    "./outputs/quantized/gptq",
    device_map="auto",
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained("./outputs/quantized/gptq")

# 추론
inputs = tokenizer("스팸 문자를 판정해주세요", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 2. vLLM 서버로 배포

```bash
# AWQ 모델로 vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/quantized/awq \
  --quantization awq \
  --dtype auto \
  --api-key token-abc123 \
  --port 8000

# GPTQ 모델로 vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/quantized/gptq \
  --quantization gptq \
  --dtype auto \
  --api-key token-abc123 \
  --port 8000
```

### 3. HuggingFace Hub에 업로드

```python
from huggingface_hub import HfApi

# 양자화된 모델 업로드
api = HfApi()
api.upload_folder(
    folder_path="./outputs/quantized/awq",
    repo_id="your-username/gemma-3-4b-spam-awq",
    repo_type="model",
)
```

## ❓ 문제 해결

### AWQ 설치 오류

**문제**: `ImportError: cannot import name 'AutoAWQForCausalLM'`

**해결**:
```bash
pip uninstall autoawq -y
pip install autoawq --extra-index-url https://wheels.autoawq.ai/
```

### CUDA Out of Memory

**문제**: 양자화 중 CUDA OOM 에러

**해결**:
```bash
# Calibration 샘플 수 줄이기
./scripts/run_quantize.sh --calibration-samples 256

# 또는 시퀀스 길이 줄이기
./scripts/run_quantize.sh --max-seq-length 1024
```

### GPTQ 양자화 느림

**문제**: GPTQ 양자화가 너무 오래 걸림

**해결**:
```bash
# Calibration 샘플 수 줄이기
./scripts/run_quantize.sh \
  --quantization-type gptq \
  --calibration-samples 256

# 또는 AWQ 사용
./scripts/run_quantize.sh --quantization-type awq
```

### 정확도 하락

**문제**: 양자화 후 성능 저하

**해결**:
```bash
# 8bit 양자화 시도
./scripts/run_quantize.sh --bits 8

# Calibration 샘플 증가
./scripts/run_quantize.sh --calibration-samples 2048

# Group size 감소 (더 정확한 양자화)
./scripts/run_quantize.sh --group-size 64
```

### 모델 로드 실패

**문제**: 양자화된 모델을 로드할 수 없음

**해결**:
```python
# AWQ: fuse_layers 비활성화
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=False
)

# GPTQ: inject_fused_attention 비활성화
model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    inject_fused_attention=False
)
```

## 🎓 추가 자료

### 참고 문서

- [AWQ 논문](https://arxiv.org/abs/2306.00978)
- [GPTQ 논문](https://arxiv.org/abs/2210.17323)
- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ GitHub](https://github.com/PanQiWei/AutoGPTQ)

### 관련 스크립트

- `scripts/run_train.sh`: 모델 학습
- `scripts/run_merge.sh`: LoRA 어댑터 병합
- `scripts/run_vllm_server.sh`: vLLM 서버 실행
- `eval/evaluation.py`: 모델 평가

## 🤝 기여

양자화 관련 개선 사항이나 버그 리포트는 이슈로 등록해주세요!

---

**Made with 🐱 by Skitty Team**

