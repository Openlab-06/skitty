# ⚙️ Configuration Module

Skitty 프로젝트의 모든 설정을 관리하는 모듈입니다.

# 파인튜닝용 패키지 설치
uv pip install axolotl
uv pip install torch torchvision torchaudio
uv pip install --no-build-isolation flash-attn
uv pip install vllm

## 📁 설정 파일 구성

| 파일 | 목적 | 설명 |
|------|------|------|
| `env_config.py` | 🌍 환경 변수 관리 | API 키, 모델명 등 민감정보 |
| `data_config.py` | 📊 데이터 처리 설정 | 중복제거, 정규화 파라미터 |
| `gemma3.yaml` | 🤖 Axolotl 학습 설정 | 모델 학습 하이퍼파라미터 |

## 🌍 환경 변수 설정 (`env_config.py`)

### 설정 클래스
```python
class ProjectConfig(BaseSettings):
    # Gemini API 설정
    GEMINI_API_KEY: str           # 필수: Gemini API 키
    GEMINI_MODEL_ARGU: str        # 데이터 증강용 모델
    GEMINI_MODEL_FILTER: str      # 데이터 필터링용 모델
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
```

### 사용법
```python
from src.config.env_config import get_config

# 싱글톤 패턴으로 설정 로드
config = get_config()
api_key = config.GEMINI_API_KEY
model_name = config.GEMINI_MODEL_ARGU
```

### 환경 파일 설정 (`.env`)
프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
# .env 파일 예시
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_ARGU=gemini-2.5-flash
GEMINI_MODEL_FILTER=gemini-2.5-flash-lite
OPENAI_API_KEY=your_gpt_api_key_here
OPENAI_MODEL=gpt-4.1-mini # 5의 경우 reasoning_effort를 low로 주어도 상대적으로 추론 시간이 길어 gpt-4.1-mini 채택
```

**⚠️ 보안 주의사항:**
- `.env` 파일은 `.gitignore`에 포함되어 있습니다
- API 키는 절대 코드에 하드코딩하지 마세요
- 환경별로 다른 `.env` 파일 사용 권장

## 📊 데이터 처리 설정 (`data_config.py`)

### 중복제거 설정
```python
class DeduplicationConfig:
    # 기본 설정
    TEXT_COL = "CN"                      # 텍스트 컬럼명
    ID_COL = "_rowid_"                   # 고유 ID 컬럼명
    
    # Simhash 파라미터
    SIMHASH_K = 3                        # 해밍 거리 임계값 (0-64)
    NGRAM_N = 2                          # Character n-gram 크기
    
    # 파일 저장 설정
    COMPRESSION = "lz4"                  # Parquet 압축 방식
    DEFAULT_UNIQUE_OUTPUT = "./src/data/deduplicated_result.parquet"
    DEFAULT_DUPS_OUTPUT = "./src/data/duplicate_analysis.parquet"
    
    # 정규화 패턴
    PHONE_PATTERN = r"0\d{1,2}-?\d{3,4}-?\d{4}"
    URL_PATTERN = r"https?://[^\s]+"
    NUM_PATTERN = r"\d{1,3}(,\d{3})+"
```

### 파라미터 설명

#### Simhash 설정
- **SIMHASH_K**: 해밍 거리 임계값
  - 낮을수록 엄격한 중복 판정 (0: 완전 일치)
  - 높을수록 관대한 중복 판정 (64: 모든 문서 중복)
  - **권장값**: 3-5 (스팸 문자 특성상)

- **NGRAM_N**: Character n-gram 크기
  - 1: 문자 단위 (너무 민감)
  - 2: 2문자 조합 (권장)
  - 3+: 긴 패턴 (덜 민감)

#### 압축 옵션
- **lz4**: 빠른 압축/해제 (권장)
- **snappy**: 압축률과 속도의 균형
- **gzip**: 높은 압축률, 느린 속도

### 사용법
```python
from src.config.data_config import DeduplicationConfig

# 설정값 사용
text_column = DeduplicationConfig.TEXT_COL
hamming_distance = DeduplicationConfig.SIMHASH_K

# CLI에서 오버라이드
python -m src.data.data_processing \
    --input data.csv \
    --text_col "message" \
    --k 5
```

## 🤖 Axolotl 학습 설정 (`gemma3.yaml`)

### 기본 모델 설정
```yaml
# 베이스 모델
base_model: google/gemma-3-4b-it
load_in_4bit: true

# 데이터셋 설정
datasets:
  - path: ./src/data/final_spam.csv
    type: alpaca
    field_instruction: instruction
    field_input: input
    field_output: output

# 출력 디렉토리
output_dir: ./outputs/gemma3
```

### LoRA 설정
```yaml
# 어댑터 설정
adapter: qlora
lora_r: 32              # LoRA rank (낮을수록 압축률 높음)
lora_alpha: 16          # LoRA scaling factor
lora_dropout: 0.05      # Dropout 비율
lora_target_modules:    # LoRA 적용 모듈
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### 학습 하이퍼파라미터
```yaml
# 시퀀스 설정
sequence_len: 512       # 최대 시퀀스 길이
sample_packing: true    # 배치 패킹 활성화

# 학습 설정
micro_batch_size: 2     # 마이크로 배치 크기
gradient_accumulation_steps: 1
num_epochs: 3           # 학습 에폭 수
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002   # 학습률

# 정규화
weight_decay: 0.0
warmup_steps: 10
evals_per_epoch: 4
saves_per_epoch: 2
```

### WandB 연동 설정
```yaml
# 실험 추적
wandb_project: skitty-spam-filter
wandb_entity: your_wandb_username
wandb_watch: gradients
wandb_name: gemma3-spam-{timestamp}
wandb_log_model: checkpoint

# 로깅 설정
logging_steps: 1
log_sweep_parameters: true
```

### 고급 설정
```yaml
# 메모리 최적화
fp16: true
bf16: false
tf32: false
gradient_checkpointing: true

# DDP 설정 (멀티 GPU)
ddp_find_unused_parameters: true
dataloader_pin_memory: false

# 안전 설정
strict: false           # 엄격 모드 비활성화
resize_token_embeddings_to_32x: true
```

## 🔧 설정 파일 사용법

### 1. 기본 학습 실행
```bash
# 기본 설정으로 학습
./run_train.sh

# 특정 설정 파일 사용
./run_train.sh custom_config.yaml
```

### 2. 설정 오버라이드
```bash
# 배치 사이즈 변경
./run_train.sh --batch-size 4

# 학습률 변경
./run_train.sh --learning-rate 0.0001

# GPU 개수 지정
./run_train.sh --gpus 2
```

### 3. 설정 검증
```bash
# 설정 파일 검증만 실행
./run_train.sh --validate-only

# 드라이 런 (명령어만 출력)
./run_train.sh --dry-run
```

## 📊 환경별 설정 관리

### 개발 환경 (Development)
```yaml
# dev_config.yaml
num_epochs: 1
micro_batch_size: 1
eval_steps: 10
save_steps: 100
wandb_mode: disabled    # WandB 비활성화
```

### 프로덕션 환경 (Production)
```yaml
# prod_config.yaml
num_epochs: 5
micro_batch_size: 4
eval_steps: 100
save_steps: 500
wandb_mode: online      # WandB 활성화
early_stopping_patience: 3
```

### 실험 환경 (Experiment)
```yaml
# exp_config.yaml
num_epochs: 10
micro_batch_size: 2
learning_rate: 0.0001
lora_r: 64             # 더 큰 LoRA rank
wandb_project: skitty-experiments
```

## 🚨 주의사항 및 베스트 프랙티스

### 1. 보안 관리
```bash
# 환경 변수 확인
echo $GEMINI_API_KEY

# API 키 유효성 검증
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     https://generativelanguage.googleapis.com/v1beta/models
```

### 2. 메모리 관리
- **4GB+ GPU**: `micro_batch_size: 2`
- **8GB+ GPU**: `micro_batch_size: 4`
- **16GB+ GPU**: `micro_batch_size: 8`

### 3. 학습 안정성
```yaml
# 학습 중단 시 재개 가능한 설정
resume_from_checkpoint: auto
save_safetensors: true
auto_resume_from_checkpoints: true
```

### 4. 디버깅 설정
```yaml
# 디버그 모드
debug: true
max_steps: 100          # 빠른 테스트
eval_steps: 10
logging_steps: 1
wandb_mode: disabled
```

## 📚 참고 자료

### Axolotl 공식 문서
- [Axolotl GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LoRA 설정 가이드](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.md)
- [데이터셋 형식](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset-formats.md)

### 모델 관련
- [Gemma-3 모델 카드](https://huggingface.co/google/gemma-3-4b-it)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)

### 모니터링
- [WandB 설정 가이드](https://docs.wandb.ai/guides/integrations/axolotl)
- [텐서보드 연동](https://pytorch.org/docs/stable/tensorboard.html)
