#!/bin/bash

# 모델 양자화 자동화 스크립트 (AWQ / GPTQ)
# 사용법: ./run_quantize.sh [옵션들]

set -e  # 오류 발생 시 스크립트 중단

# 기본 설정
DEFAULT_MODEL_PATH="./outputs/merged"
DEFAULT_OUTPUT_DIR="./outputs/quantized"
DEFAULT_QUANTIZATION_TYPE="awq"
DEFAULT_BITS=4
DEFAULT_GROUP_SIZE=128
DEFAULT_CALIBRATION_SAMPLES=512

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
    echo "${CYAN}🚀 모델 양자화 자동화 스크립트 (AWQ / GPTQ) 🚀${NC}"
    echo ""
    echo "${BLUE}사용법:${NC}"
    echo "  $0 [옵션들]"
    echo ""
    echo "${GREEN}예시:${NC}"
    echo "  $0                                          # 기본 설정(AWQ)으로 양자화"
    echo "  $0 --quantization-type gptq                 # GPTQ로 양자화"
    echo "  $0 --model-path ./outputs/merged --bits 8   # 8bit 양자화"
    echo "  $0 --dry-run                                # 명령어만 출력 (실제 실행 안함)"
    echo ""
    echo "${YELLOW}필수 옵션:${NC}"
    echo "  --model-path PATH                           양자화할 모델 경로 (기본값: $DEFAULT_MODEL_PATH)"
    echo "  --output-dir DIR                            출력 디렉토리 (기본값: $DEFAULT_OUTPUT_DIR)"
    echo ""
    echo "${YELLOW}양자화 설정:${NC}"
    echo "  --quantization-type TYPE                    양자화 타입: awq 또는 gptq (기본값: $DEFAULT_QUANTIZATION_TYPE)"
    echo "  --bits N                                    양자화 비트 수: 2, 3, 4, 8 (기본값: $DEFAULT_BITS)"
    echo "  --group-size N                              그룹 사이즈 (기본값: $DEFAULT_GROUP_SIZE)"
    echo ""
    echo "${YELLOW}Calibration 데이터 설정:${NC}"
    echo "  --calibration-samples N                     Calibration 샘플 수 (기본값: $DEFAULT_CALIBRATION_SAMPLES)"
    echo "  --dataset-name NAME                         데이터셋 이름 (기본값: Devocean-06/Spam_QA-Corpus)"
    echo "  --dataset-split SPLIT                       데이터셋 스플릿 (기본값: train)"
    echo "  --max-seq-length N                          최대 시퀀스 길이 (기본값: 1500)"
    echo ""
    echo "${YELLOW}기타 옵션:${NC}"
    echo "  -h, --help                                  이 도움말 출력"
    echo "  --dry-run                                   실제 실행 없이 명령어만 출력"
    echo "  --verbose                                   상세한 로그 출력"
    echo ""
    echo "${CYAN}📚 양자화 타입 설명:${NC}"
    echo "  ${GREEN}AWQ (Activation-aware Weight Quantization)${NC}"
    echo "    - 더 빠른 추론 속도"
    echo "    - 정확도 손실이 적음"
    echo "    - 권장: 대부분의 경우"
    echo ""
    echo "  ${GREEN}GPTQ (Generative Pre-trained Transformer Quantization)${NC}"
    echo "    - 안정적인 양자화"
    echo "    - 광범위한 하드웨어 지원"
    echo "    - 권장: AWQ가 작동하지 않을 때"
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
        log_warning "양자화는 가능하지만 시간이 오래 걸릴 수 있습니다."
    fi
    
    # 파이썬 환경 확인
    if command -v python &> /dev/null; then
        local python_version=$(python --version 2>&1 | cut -d' ' -f2)
        log_info "Python 버전: $python_version"
    else
        log_error "Python이 설치되어 있지 않습니다."
        exit 1
    fi
    
    # 메모리 확인
    local total_memory=$(free -h | awk '/^Mem:/ {print $2}')
    local available_memory=$(free -h | awk '/^Mem:/ {print $7}')
    log_info "시스템 메모리: 총 ${total_memory}, 사용 가능: ${available_memory}"
}

# 함수: 모델 경로 검증
validate_model_path() {
    local model_path="$1"
    
    log_step "모델 경로 검증"
    
    if [[ ! -d "$model_path" ]]; then
        log_error "모델 디렉토리를 찾을 수 없습니다: $model_path"
        log_info "먼저 모델을 merge 해야 합니다:"
        log_info "  ./scripts/run_merge.sh"
        exit 1
    fi
    
    # config.json 또는 model 파일 확인
    if [[ ! -f "$model_path/config.json" ]]; then
        log_error "모델 설정 파일(config.json)을 찾을 수 없습니다: $model_path"
        exit 1
    fi
    
    log_success "모델 경로 확인: $model_path"
}

# 함수: 출력 디렉토리 준비
prepare_output_dir() {
    local output_dir="$1"
    local quantization_type="$2"
    
    log_step "출력 디렉토리 준비"
    
    # 양자화 타입별 서브 디렉토리 생성
    local full_output_dir="${output_dir}/${quantization_type}"
    
    if [[ -d "$full_output_dir" ]]; then
        log_warning "출력 디렉토리가 이미 존재합니다: $full_output_dir"
        log_warning "기존 파일을 덮어쓸 수 있습니다."
    else
        mkdir -p "$full_output_dir"
        log_info "출력 디렉토리 생성: $full_output_dir"
    fi
    
    echo "$full_output_dir"
}

# 함수: 양자화 실행
run_quantization() {
    local model_path="$1"
    local output_dir="$2"
    local quantization_type="$3"
    local bits="$4"
    local group_size="$5"
    local calibration_samples="$6"
    local dataset_name="$7"
    local dataset_split="$8"
    local max_seq_length="$9"
    local dry_run="${10}"
    local verbose="${11}"
    
    log_step "양자화 시작"
    
    # 명령어 구성
    local cmd="python -m src.optimizer.quantize"
    cmd="$cmd --model-path \"$model_path\""
    cmd="$cmd --output-dir \"$output_dir\""
    cmd="$cmd --quantization-type $quantization_type"
    cmd="$cmd --bits $bits"
    cmd="$cmd --group-size $group_size"
    cmd="$cmd --calibration-samples $calibration_samples"
    
    if [[ -n "$dataset_name" ]]; then
        cmd="$cmd --dataset-name \"$dataset_name\""
    fi
    
    if [[ -n "$dataset_split" ]]; then
        cmd="$cmd --dataset-split $dataset_split"
    fi
    
    if [[ -n "$max_seq_length" ]]; then
        cmd="$cmd --max-seq-length $max_seq_length"
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN - 실행될 명령어:"
        echo "  $cmd"
        return 0
    fi
    
    log_info "양자화 설정:"
    log_info "  - 타입: ${quantization_type^^}"
    log_info "  - 비트: ${bits}bit"
    log_info "  - 그룹 사이즈: $group_size"
    log_info "  - Calibration 샘플: $calibration_samples"
    log_info ""
    log_info "실행 명령어: $cmd"
    
    # 실행 시간 측정
    local start_time=$(date +%s)
    
    echo ""
    echo "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo "${CYAN}                 ⚡ 양자화 시작                         ${NC}"
    echo "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # 명령어 실행
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        echo ""
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        echo "${GREEN}                 🎉 양자화 완료!                       ${NC}"
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        log_success "총 소요시간: ${hours}시간 ${minutes}분 ${seconds}초"
        log_info "양자화된 모델 경로: $output_dir"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo ""
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        echo "${RED}                 ❌ 양자화 실패                        ${NC}"
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        log_error "양자화 중 오류가 발생했습니다. (소요시간: ${duration}초)"
        exit 1
    fi
}

# 메인 함수
main() {
    local model_path="$DEFAULT_MODEL_PATH"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local quantization_type="$DEFAULT_QUANTIZATION_TYPE"
    local bits="$DEFAULT_BITS"
    local group_size="$DEFAULT_GROUP_SIZE"
    local calibration_samples="$DEFAULT_CALIBRATION_SAMPLES"
    local dataset_name=""
    local dataset_split=""
    local max_seq_length=""
    local dry_run="false"
    local verbose="false"
    
    # 인자 파싱
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --model-path)
                model_path="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --quantization-type)
                quantization_type="$2"
                shift 2
                ;;
            --bits)
                bits="$2"
                shift 2
                ;;
            --group-size)
                group_size="$2"
                shift 2
                ;;
            --calibration-samples)
                calibration_samples="$2"
                shift 2
                ;;
            --dataset-name)
                dataset_name="$2"
                shift 2
                ;;
            --dataset-split)
                dataset_split="$2"
                shift 2
                ;;
            --max-seq-length)
                max_seq_length="$2"
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
            -*)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
            *)
                log_error "알 수 없는 인자: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 시스템 환경 확인
    check_system_requirements
    
    # 모델 경로 검증
    validate_model_path "$model_path"
    
    # 출력 디렉토리 준비
    local full_output_dir=$(prepare_output_dir "$output_dir" "$quantization_type")
    
    # 양자화 실행
    run_quantization \
        "$model_path" \
        "$full_output_dir" \
        "$quantization_type" \
        "$bits" \
        "$group_size" \
        "$calibration_samples" \
        "$dataset_name" \
        "$dataset_split" \
        "$max_seq_length" \
        "$dry_run" \
        "$verbose"
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
echo "${PURPLE}          🐱 모델 양자화 자동화 플랫폼 🐱             ${NC}"
echo ""

# 메인 함수 실행
main "$@"

