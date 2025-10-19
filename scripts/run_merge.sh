#!/usr/bin/env bash

# Axolotl LoRA 모델 병합 자동화 스크립트
# 사용법: ./run_merge.sh [설정파일] [옵션들]

set -e  # 오류 발생 시 스크립트 중단

# 기본 설정
DEFAULT_CONFIG="./src/config/gemma3.yaml"
DEFAULT_LORA_MODEL_DIR="./outputs/gemma3/checkpoint-85"
DEFAULT_LOG_DIR="./logs"

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
PINK='\033[1;35m'
NC='\033[0m' # No Color

# 함수: 도움말 출력
show_help() {
    echo ""
    echo "${CYAN}🚀 Axolotl LoRA 모델 병합 자동화 스크립트 🚀${NC}"
    echo ""
    echo "${BLUE}사용법:${NC}"
    echo "  $0 [설정파일] [옵션들]"
    echo ""
    echo "${GREEN}예시:${NC}"
    echo "  $0                                    # 기본 설정으로 병합 시작"
    echo "  $0 custom_config.yaml                 # 특정 설정 파일로 병합"
    echo "  $0 --dry-run                          # 명령어만 출력 (실제 실행 안함)"
    echo ""
    echo "${YELLOW}옵션:${NC}"
    echo "  -h, --help                            이 도움말 출력"
    echo "  --config CONFIG                       설정 파일 경로 지정"
    echo "  --lora-model-dir DIR                  LoRA 모델 디렉토리 (기본값: $DEFAULT_LORA_MODEL_DIR)"
    echo "  --log-dir DIR                         로그 디렉토리 (기본값: $DEFAULT_LOG_DIR)"
    echo "  --dry-run                             실제 실행 없이 명령어만 출력"
    echo "  --verbose                             상세한 로그 출력"
    echo "  --quiet                               최소한의 로그만 출력"
    echo ""
    echo "${CYAN}📝 참고:${NC}"
    echo "  병합된 모델은 config 파일의 output_dir 아래 merged/ 폴더에 저장됩니다."
    echo "  예: output_dir이 './outputs/gemma3'이면 './outputs/gemma3/merged'에 저장"
    echo ""
}

# 함수: 로그 출력
log_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo "${PURPLE}➤ $1${NC}"
    echo "$(printf '%*s' 50 | tr ' ' '-')"
}

# 함수: 시스템 환경 확인
check_system_requirements() {
    log_step "시스템 환경 확인"
    
    # Hugging Face 토큰 확인
    if [ -z "$HF_TOKEN" ]; then
        log_warning "HF_TOKEN 환경 변수가 설정되지 않았습니다."
        log_info "Hugging Face 토큰을 설정하려면:"
        log_info "  export HF_TOKEN='your_token_here'"
        log_info "또는 ~/.huggingface/token 파일에 토큰을 저장하세요."
    else
        log_success "Hugging Face 토큰이 감지되었습니다."
    fi
    
    # CUDA 확인
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        log_info "GPU 감지: ${gpu_count}개 GPU, 메모리: ${gpu_memory}MB"
        
        # GPU 사용률 확인
        log_info "현재 GPU 상태:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
        while read line; do
            echo "    $line"
        done
    else
        log_warning "NVIDIA GPU가 감지되지 않았습니다."
    fi
    
    # 파이썬 환경 확인
    if command -v python &> /dev/null; then
        local python_version=$(python --version 2>&1 | cut -d' ' -f2)
        log_info "Python 버전: $python_version"
    else
        log_error "Python이 설치되어 있지 않습니다."
        exit 1
    fi
    
    # Axolotl 설치 확인
    if python -c "import axolotl" 2>/dev/null; then
        log_success "Axolotl 패키지 확인 완료"
    else
        log_error "Axolotl이 설치되어 있지 않습니다."
        log_info "다음 명령어로 설치하세요: pip install axolotl"
        exit 1
    fi
    
    # 메모리 확인
    local total_memory=$(free -h | awk '/^Mem:/ {print $2}')
    local available_memory=$(free -h | awk '/^Mem:/ {print $7}')
    log_info "시스템 메모리: 총 ${total_memory}, 사용 가능: ${available_memory}"
}

# 함수: 설정 파일 검증
validate_config() {
    local config_file="$1"
    
    log_step "설정 파일 검증"
    
    if [ ! -f "$config_file" ]; then
        log_error "설정 파일을 찾을 수 없습니다: $config_file"
        exit 1
    fi
    
    log_info "설정 파일: $config_file"
    
    # YAML 문법 확인
    if command -v python &> /dev/null; then
        if python -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
            log_success "YAML 문법 검증 완료"
        else
            log_error "YAML 파일 형식이 올바르지 않습니다."
            exit 1
        fi
    fi
    
    # 주요 설정 확인
    log_info "주요 설정 내용:"
    if command -v grep &> /dev/null && command -v awk &> /dev/null; then
        echo "    모델명: $(grep -E '^base_model:' "$config_file" | awk '{print $2}' || echo 'N/A')"
        echo "    학습률: $(grep -E '^learning_rate:' "$config_file" | awk '{print $2}' || echo 'N/A')"
        echo "    에포크: $(grep -E '^num_epochs:' "$config_file" | awk '{print $2}' || echo 'N/A')"
    fi
}

# 함수: 디렉토리 준비
prepare_directories() {
    local lora_model_dir="$1"
    local log_dir="$2"
    local config_file="$3"
    
    log_step "디렉토리 준비"
    
    # LoRA 모델 디렉토리 확인
    if [ ! -d "$lora_model_dir" ]; then
        log_error "LoRA 모델 디렉토리를 찾을 수 없습니다: $lora_model_dir"
        exit 1
    fi
    log_info "LoRA 모델 디렉토리 확인: $lora_model_dir"
    
    # 로그 디렉토리 생성
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
        log_info "로그 디렉토리 생성: $log_dir"
    fi
    log_info "로그 디렉토리 확인: $log_dir"
    
    # config 파일에서 output_dir 읽기
    local config_output_dir=$(grep -E '^output_dir:' "$config_file" | awk '{print $2}' 2>/dev/null || echo "")
    if [ -n "$config_output_dir" ]; then
        log_info "병합된 모델 저장 경로: ${config_output_dir}/merged"
    else
        log_info "병합된 모델은 config 파일의 output_dir/merged에 저장됩니다."
    fi
}

# 함수: LoRA 병합 실행
run_merge() {
    local config_file="$1"
    local lora_model_dir="$2"
    local dry_run="$3"
    local log_dir="$4"
    local verbose="$5"
    
    log_step "Axolotl LoRA 병합 시작"
    
    # 기본 명령어 구성
    local cmd="axolotl merge-lora \"$config_file\" --lora-model-dir=\"$lora_model_dir\""
    
    # 로그 설정
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$log_dir/axolotl_merge_${timestamp}.log"
    
    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi
    
    if [ "$dry_run" = "true" ]; then
        log_info "DRY RUN - 실행될 명령어:"
        echo "  $cmd"
        echo "  로그 파일: $log_file"
        return 0
    fi
    
    log_info "실행 명령어: $cmd"
    log_info "로그 파일: $log_file"
    
    # 실행 시간 측정
    local start_time=$(date +%s)
    
    echo ""
    echo "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo "${CYAN}                    🎯 LoRA 병합 시작                    ${NC}"
    echo "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # 명령어 실행 (로그 파일에도 저장)
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        echo ""
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        echo "${GREEN}                    🎉 LoRA 병합 완료!                  ${NC}"
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        log_success "총 소요시간: ${hours}시간 ${minutes}분 ${seconds}초"
        log_info "로그 파일: $log_file"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo ""
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        echo "${RED}                    ❌ LoRA 병합 실패                    ${NC}"
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        log_error "LoRA 병합 중 오류가 발생했습니다. (소요시간: ${duration}초)"
        log_error "상세한 로그는 다음 파일을 확인하세요: $log_file"
        exit 1
    fi
}

# 메인 함수
main() {
    local config_file="$DEFAULT_CONFIG"
    local lora_model_dir="$DEFAULT_LORA_MODEL_DIR"
    local log_dir="$DEFAULT_LOG_DIR"
    local dry_run="false"
    local verbose="false"
    local quiet="false"
    
    # 인자 파싱
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            --config)
                config_file="$2"
                shift 2
                ;;
            --lora-model-dir)
                lora_model_dir="$2"
                shift 2
                ;;
            --log-dir)
                log_dir="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --verbose)
                verbose="true"
                shift
                ;;
            --quiet)
                quiet="true"
                shift
                ;;
            *)
                # 첫 번째 위치 인자는 설정 파일로 처리
                if [ "$config_file" = "$DEFAULT_CONFIG" ]; then
                    config_file="$1"
                else
                    log_error "너무 많은 인자입니다: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Quiet 모드가 아닌 경우에만 시스템 확인
    if [ "$quiet" != "true" ]; then
        check_system_requirements
    fi
    
    # 설정 파일 검증
    validate_config "$config_file"
    
    # 디렉토리 준비
    prepare_directories "$lora_model_dir" "$log_dir" "$config_file"
    
    # LoRA 병합 실행
    run_merge "$config_file" "$lora_model_dir" "$dry_run" "$log_dir" "$verbose"
}

# 스크립트 시작
echo ""
echo "${PURPLE}        ∩───∩        ${NC}"
echo "${PURPLE}       (  ◕   ◕ )      ${NC}" 
echo "${PURPLE}      /           \\     ${NC}"
echo "${PURPLE}     (  ~~~   ~~~  )    ${NC}"
echo "${PURPLE}      \\___________/     ${NC}"
echo "${PINK}         ∪     ∪        ${NC}"
echo ""
echo "${CYAN}███████╗██╗  ██╗██╗████████╗████████╗██╗   ██╗${NC}"
echo "${CYAN}██╔════╝██║ ██╔╝██║╚══██╔══╝╚══██╔══╝╚██╗ ██╔╝${NC}"
echo "${CYAN}███████╗█████╔╝ ██║   ██║      ██║    ╚████╔╝ ${NC}"
echo "${CYAN}╚════██║██╔═██╗ ██║   ██║      ██║     ╚██╔╝  ${NC}"
echo "${CYAN}███████║██║  ██╗██║   ██║      ██║      ██║   ${NC}"
echo "${CYAN}╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝      ╚═╝   ${NC}"
echo ""
echo "${PURPLE}          🐱 LoRA 모델 병합 자동화 플랫폼 🐱           ${NC}"
echo ""

# 메인 함수 실행
main "$@"
