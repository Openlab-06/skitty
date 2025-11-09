#!/usr/bin/env bash

# Axolotl 모델 학습 자동화 스크립트
# 사용법: ./train_axolotl.sh [설정파일] [옵션들]

set -e  # 오류 발생 시 스크립트 중단

# 스크립트 디렉토리 기준으로 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 기본 설정
DEFAULT_CONFIG="$PROJECT_ROOT/src/config/gemma3.yaml"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs"
DEFAULT_LOG_DIR="$PROJECT_ROOT/logs"

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
    echo "${CYAN}🚀 Axolotl 모델 학습 자동화 스크립트 🚀${NC}"
    echo ""
    echo "${BLUE}사용법:${NC}"
    echo "  $0 [설정파일] [옵션들]"
    echo ""
    echo "${GREEN}예시:${NC}"
    echo "  $0                                    # 기본 설정으로 학습 시작"
    echo "  $0 custom_config.yaml                 # 특정 설정 파일로 학습"
    echo "  $0 --resume                           # 중단된 학습 재개"
    echo "  $0 --validate-only                    # 설정 검증만 실행"
    echo "  $0 --preprocess-only                  # 데이터 전처리만 실행"
    echo "  $0 --dry-run                          # 명령어만 출력 (실제 실행 안함)"
    echo ""
    echo "${YELLOW}옵션:${NC}"
    echo "  -h, --help                            이 도움말 출력"
    echo "  --resume                              중단된 학습 재개"
    echo "  --validate-only                       설정 파일 검증만 실행"
    echo "  --preprocess-only                     데이터 전처리만 실행"
    echo "  --inference                           추론 모드로 실행"
    echo "  --config CONFIG                       설정 파일 경로 지정"
    echo "  --output-dir DIR                      출력 디렉토리 (기본값: $DEFAULT_OUTPUT_DIR)"
    echo "  --log-dir DIR                         로그 디렉토리 (기본값: $DEFAULT_LOG_DIR)"
    echo "  --gpus N                              사용할 GPU 개수 지정"
    echo "  --batch-size N                        배치 사이즈 오버라이드"
    echo "  --learning-rate LR                    학습률 오버라이드"
    echo "  --epochs N                            에포크 수 오버라이드"
    echo "  --dry-run                             실제 실행 없이 명령어만 출력"
    echo "  --verbose                             상세한 로그 출력"
    echo "  --quiet                               최소한의 로그만 출력"
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
        log_warning "NVIDIA GPU가 감지되지 않았습니다. CPU 학습이 진행될 수 있습니다."
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
        echo "    데이터셋: $(grep -E '^datasets:' -A 5 "$config_file" | grep -E 'path:' | head -1 | awk '{print $2}' || echo 'N/A')"
        echo "    출력 디렉토리: $(grep -E '^output_dir:' "$config_file" | awk '{print $2}' || echo 'N/A')"
        echo "    학습률: $(grep -E '^learning_rate:' "$config_file" | awk '{print $2}' || echo 'N/A')"
        echo "    에포크: $(grep -E '^num_epochs:' "$config_file" | awk '{print $2}' || echo 'N/A')"
    fi
}

# 함수: 디렉토리 준비
prepare_directories() {
    local output_dir="$1"
    local log_dir="$2"
    
    log_step "디렉토리 준비"
    
    if [ ! -d "$output_dir" ]; then
        log_info "출력 디렉토리를 생성합니다: $output_dir"
        mkdir -p "$output_dir"
    fi
    log_info "출력 디렉토리 확인: $output_dir"
    
    if [ ! -d "$log_dir" ]; then
        log_info "로그 디렉토리를 생성합니다: $log_dir"
        mkdir -p "$log_dir"
    fi
    log_info "로그 디렉토리 확인: $log_dir"
}

# 함수: 학습 실행
run_training() {
    local config_file="$1"
    local mode="$2"
    local dry_run="$3"
    local additional_args="$4"
    local log_dir="$5"
    local verbose="$6"
    
    log_step "Axolotl 학습 시작"
    
    # 기본 명령어 구성
    local cmd="axolotl train \"$config_file\""
    
    # 모드별 옵션 추가
    case "$mode" in
        "resume")
            cmd="$cmd --resume_from_checkpoint"
            log_info "중단된 학습을 재개합니다..."
            ;;
        "validate")
            cmd="axolotl validate \"$config_file\""
            log_info "설정 파일 검증을 실행합니다..."
            ;;
        "preprocess")
            cmd="axolotl preprocess \"$config_file\""
            log_info "데이터 전처리를 실행합니다..."
            ;;
        "inference")
            cmd="axolotl inference \"$config_file\""
            log_info "추론 모드로 실행합니다..."
            ;;
        "train")
            log_info "모델 학습을 시작합니다..."
            ;;
    esac
    
    # 추가 인자 적용
    if [ -n "$additional_args" ]; then
        cmd="$cmd $additional_args"
    fi
    
    # 로그 설정
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$log_dir/axolotl_${mode}_${timestamp}.log"
    
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
    echo "${CYAN}                    🎯 학습 시작                       ${NC}"
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
        echo "${GREEN}                    🎉 학습 완료!                      ${NC}"
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        log_success "총 소요시간: ${hours}시간 ${minutes}분 ${seconds}초"
        log_info "로그 파일: $log_file"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo ""
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        echo "${RED}                    ❌ 학습 실패                       ${NC}"
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        log_error "학습 중 오류가 발생했습니다. (소요시간: ${duration}초)"
        log_error "상세한 로그는 다음 파일을 확인하세요: $log_file"
        exit 1
    fi
}

# 메인 함수
main() {
    local config_file="$DEFAULT_CONFIG"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local log_dir="$DEFAULT_LOG_DIR"
    local mode="train"
    local dry_run="false"
    local additional_args=""
    local verbose="false"
    local quiet="false"
    
    # 인자 파싱
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            --resume)
                mode="resume"
                shift
                ;;
            --validate-only)
                mode="validate"
                shift
                ;;
            --preprocess-only)
                mode="preprocess"
                shift
                ;;
            --inference)
                mode="inference"
                shift
                ;;
            --config)
                config_file="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --log-dir)
                log_dir="$2"
                shift 2
                ;;
            --gpus)
                additional_args="$additional_args --gpus $2"
                shift 2
                ;;
            --batch-size)
                additional_args="$additional_args --batch_size $2"
                shift 2
                ;;
            --learning-rate)
                additional_args="$additional_args --learning_rate $2"
                shift 2
                ;;
            --epochs)
                additional_args="$additional_args --num_epochs $2"
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
            -*)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
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
    if [ "$quiet" != "true" ] && [ "$mode" != "validate" ]; then
        check_system_requirements
    fi
    
    # 설정 파일 검증
    validate_config "$config_file"
    
    # 디렉토리 준비
    if [ "$mode" != "validate" ]; then
        prepare_directories "$output_dir" "$log_dir"
    fi
    
    # 학습 실행
    run_training "$config_file" "$mode" "$dry_run" "$additional_args" "$log_dir" "$verbose"
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
echo "${PURPLE}          🐱 AI 모델 학습 자동화 플랫폼 🐱          ${NC}"
echo ""

# 메인 함수 실행
main "$@"