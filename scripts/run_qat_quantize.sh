#!/usr/bin/env bash

# Axolotl QAT 모델 양자화 스크립트
# QAT로 학습한 모델을 양자화하여 최종 모델 생성
# 사용법: ./run_qat_quantize.sh [설정파일]

set -e  # 오류 발생 시 스크립트 중단

# 스크립트 디렉토리 기준으로 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 기본 설정
DEFAULT_CONFIG="$PROJECT_ROOT/src/config/gemma3-qat.yaml"
DEFAULT_VENV="$PROJECT_ROOT/.venv311"
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
    echo "${CYAN}🚀 Axolotl QAT 모델 양자화 스크립트 🚀${NC}"
    echo ""
    echo "${BLUE}사용법:${NC}"
    echo "  $0 [설정파일] [옵션들]"
    echo ""
    echo "${GREEN}예시:${NC}"
    echo "  $0                                    # 기본 설정으로 QAT 모델 양자화"
    echo "  $0 gemma3-qat-int8.yaml               # 특정 설정 파일로 양자화"
    echo "  $0 --config src/config/custom.yaml    # 설정 파일 명시"
    echo "  $0 --dry-run                          # 명령어만 출력 (실제 실행 안함)"
    echo ""
    echo "${YELLOW}옵션:${NC}"
    echo "  -h, --help                            이 도움말 출력"
    echo "  --config CONFIG                       설정 파일 경로 지정 (기본값: $DEFAULT_CONFIG)"
    echo "  --venv PATH                           가상환경 경로 (기본값: $DEFAULT_VENV)"
    echo "  --log-dir DIR                         로그 디렉토리 (기본값: $DEFAULT_LOG_DIR)"
    echo "  --dry-run                             실제 실행 없이 명령어만 출력"
    echo "  --verbose                             상세한 로그 출력"
    echo ""
    echo "${CYAN}📚 QAT 양자화 프로세스:${NC}"
    echo "  ${GREEN}1단계: QAT 학습${NC}"
    echo "    - axolotl train으로 QAT 모델 학습"
    echo "    - 학습 시 fp8/int8 등의 양자화 인식 학습 수행"
    echo ""
    echo "  ${GREEN}2단계: 양자화 (현재 스크립트)${NC}"
    echo "    - axolotl quantize로 학습된 모델 양자화"
    echo "    - output_dir/quantized 디렉토리에 최종 모델 저장"
    echo ""
    echo "${YELLOW}⚠️  주의사항:${NC}"
    echo "  - QAT 학습이 먼저 완료되어야 합니다"
    echo "  - 학습에 사용한 동일한 config 파일을 사용해야 합니다"
    echo "  - 양자화된 모델은 output_dir/quantized에 저장됩니다"
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

# 함수: 가상환경 확인 및 활성화
activate_venv() {
    local venv_path="$1"

    log_step "가상환경 확인 및 활성화"

    if [ ! -d "$venv_path" ]; then
        log_error "가상환경을 찾을 수 없습니다: $venv_path"
        exit 1
    fi

    log_info "가상환경 경로: $venv_path"

    # 활성화 스크립트 확인
    if [ ! -f "$venv_path/bin/activate" ]; then
        log_error "가상환경 활성화 스크립트를 찾을 수 없습니다: $venv_path/bin/activate"
        exit 1
    fi

    log_success "가상환경 확인 완료"
}

# 함수: Axolotl 설치 확인
check_axolotl() {
    local venv_path="$1"

    log_step "Axolotl 패키지 확인"

    if "$venv_path/bin/python" -c "import axolotl" 2>/dev/null; then
        local axolotl_version=$("$venv_path/bin/python" -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
        log_success "Axolotl 설치 확인 완료 (버전: $axolotl_version)"
    else
        log_error "Axolotl이 설치되어 있지 않습니다."
        log_info "다음 명령어로 설치하세요: pip install axolotl"
        exit 1
    fi
}

# 함수: 설정 파일 검증
validate_config() {
    local config_file="$1"
    local venv_path="$2"

    log_step "설정 파일 검증"

    if [ ! -f "$config_file" ]; then
        log_error "설정 파일을 찾을 수 없습니다: $config_file"
        exit 1
    fi

    log_info "설정 파일: $config_file"

    # YAML 문법 확인
    if "$venv_path/bin/python" -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
        log_success "YAML 문법 검증 완료"
    else
        log_error "YAML 파일 형식이 올바르지 않습니다."
        exit 1
    fi

    # output_dir 확인
    local output_dir=$(grep -E '^output_dir:' "$config_file" | awk '{print $2}' | sed 's/^"\(.*\)"$/\1/' || echo '')

    # 상대 경로를 절대 경로로 변환
    if [[ "$output_dir" == ./* ]] || [[ "$output_dir" == ../* ]]; then
        output_dir="$PROJECT_ROOT/${output_dir#./}"
    fi

    if [ -z "$output_dir" ]; then
        log_error "설정 파일에 output_dir이 지정되지 않았습니다."
        exit 1
    fi

    if [ ! -d "$output_dir" ]; then
        log_error "QAT 모델 디렉토리를 찾을 수 없습니다: $output_dir"
        log_error "먼저 QAT 학습을 완료해주세요. (./scripts/run_qat.sh)"
        exit 1
    fi

    log_success "QAT 모델 디렉토리 확인: $output_dir"

    # 주요 설정 확인
    log_info "주요 설정 내용:"
    echo "    모델명: $(grep -E '^base_model:' "$config_file" | awk '{print $2}' || echo 'N/A')"
    echo "    출력 디렉토리: $output_dir"
    echo "    양자화 타입: $(grep -E '^weight_dtype:' "$config_file" | awk '{print $2}' || echo 'N/A')"
    echo "    Activation dtype: $(grep -E '^activation_dtype:' "$config_file" | awk '{print $2}' || echo 'N/A')"
}

# 함수: 디렉토리 준비
prepare_directories() {
    local log_dir="$1"

    log_step "디렉토리 준비"

    if [ ! -d "$log_dir" ]; then
        log_info "로그 디렉토리를 생성합니다: $log_dir"
        mkdir -p "$log_dir"
    fi
    log_info "로그 디렉토리 확인: $log_dir"
}

# 함수: 양자화 실행
run_quantization() {
    local config_file="$1"
    local venv_path="$2"
    local log_dir="$3"
    local dry_run="$4"
    local verbose="$5"

    log_step "QAT 모델 양자화 시작"

    # 명령어 구성
    local cmd="source \"$venv_path/bin/activate\" && axolotl quantize \"$config_file\""

    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi

    # 로그 파일 설정
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$log_dir/axolotl_qat_quantize_${timestamp}.log"

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
    echo "${CYAN}               ⚡ QAT 모델 양자화 시작 ⚡              ${NC}"
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
        echo "${GREEN}               🎉 양자화 완료! 🎉                     ${NC}"
        echo "${GREEN}════════════════════════════════════════════════════════${NC}"
        log_success "총 소요시간: ${hours}시간 ${minutes}분 ${seconds}초"
        log_info "로그 파일: $log_file"

        # output_dir 확인하여 quantized 디렉토리 경로 출력
        local output_dir=$(grep -E '^output_dir:' "$config_file" | awk '{print $2}' | sed 's/^"\(.*\)"$/\1/' || echo '')
        if [[ "$output_dir" == ./* ]] || [[ "$output_dir" == ../* ]]; then
            output_dir="$PROJECT_ROOT/${output_dir#./}"
        fi

        log_success "양자화된 모델 경로: ${output_dir}/quantized"

        echo ""
        log_info "다음 단계:"
        echo "  1. 양자화된 모델 테스트: axolotl inference ${config_file}"
        echo "  2. vLLM 서버로 배포: ./scripts/run_vllm_server.sh --model-path ${output_dir}/quantized"
        echo "  3. HuggingFace에 업로드: ./scripts/run_upload.sh --model-path ${output_dir}/quantized"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo ""
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        echo "${RED}               ❌ 양자화 실패 ❌                      ${NC}"
        echo "${RED}════════════════════════════════════════════════════════${NC}"
        log_error "양자화 중 오류가 발생했습니다. (소요시간: ${duration}초)"
        log_error "상세한 로그는 다음 파일을 확인하세요: $log_file"
        exit 1
    fi
}

# 메인 함수
main() {
    local config_file="$DEFAULT_CONFIG"
    local venv_path="$DEFAULT_VENV"
    local log_dir="$DEFAULT_LOG_DIR"
    local dry_run="false"
    local verbose="false"

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
            --venv)
                venv_path="$2"
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
            -*)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
            *)
                # 첫 번째 위치 인자는 설정 파일로 처리
                if [ "$config_file" = "$DEFAULT_CONFIG" ]; then
                    # 상대 경로를 절대 경로로 변환
                    if [[ "$1" != /* ]]; then
                        config_file="$PROJECT_ROOT/$1"
                    else
                        config_file="$1"
                    fi
                else
                    log_error "너무 많은 인자입니다: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # 가상환경 확인 및 활성화
    activate_venv "$venv_path"

    # Axolotl 설치 확인
    check_axolotl "$venv_path"

    # 설정 파일 검증
    validate_config "$config_file" "$venv_path"

    # 디렉토리 준비
    prepare_directories "$log_dir"

    # 양자화 실행
    run_quantization "$config_file" "$venv_path" "$log_dir" "$dry_run" "$verbose"
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
echo "${PURPLE}       🐱 QAT 모델 양자화 자동화 🐱                ${NC}"
echo ""

# 메인 함수 실행
main "$@"
