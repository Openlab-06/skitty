#!/bin/bash

# 데이터 처리 파이프라인 자동화 스크립트
# 사용법: ./run_data_pipeline.sh [입력파일] [옵션들]

set -e  # 오류 발생 시 스크립트 중단

# 기본 설정
DEFAULT_INPUT_FILE="./src/data/raw_spam_2025.csv"
DEFAULT_TEXT_COL="CN"
DEFAULT_OUTPUT_DIR="./src/data"

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수: 도움말 출력
show_help() {
    echo "데이터 처리 파이프라인 자동화 스크립트"
    echo ""
    echo "사용법:"
    echo "  $0 [입력파일] [옵션들]"
    echo ""
    echo "예시:"
    echo "  $0                                    # 기본 설정으로 전체 파이프라인 실행"
    echo "  $0 my_data.csv                        # 특정 파일로 전체 파이프라인 실행"
    echo "  $0 --dedup-only                       # 중복제거만 실행"
    echo "  $0 --filter-only                      # 데이터 필터링만 실행"
    echo "  $0 --aug-only                         # 데이터 증강만 실행"
    echo "  $0 --status                           # 파이프라인 상태 확인"
    echo "  $0 --sample-size 0.05 --sample-seed 123  # 5% 샘플링, 시드 123"
    echo "  $0 --filter-only --sample-size 0.1   # 필터링만 실행, 10% 샘플링"
    echo ""
    echo "옵션:"
    echo "  -h, --help                            이 도움말 출력"
    echo "  --dedup-only                          중복제거만 실행"
    echo "  --filter-only                         데이터 필터링만 실행"
    echo "  --aug-only                            데이터 증강만 실행"
    echo "  --status                              파이프라인 상태 확인"
    echo "  --skip-dedup                          중복제거 건너뛰기"
    echo "  --skip-filter                         데이터 필터링 건너뛰기"
    echo "  --skip-aug                            데이터 증강 건너뛰기"
    echo "  --text-col COLUMN                     텍스트 컬럼명 (기본값: $DEFAULT_TEXT_COL)"
    echo "  --output-dir DIR                      출력 디렉토리 (기본값: $DEFAULT_OUTPUT_DIR)"
    echo "  --sample-size FLOAT                   필터링용 샘플링 비율 (예: 0.02 = 2%)"
    echo "  --sample-seed INT                     샘플링 시드 (예: 42)"
    echo "  --dry-run                             실제 실행 없이 명령어만 출력"
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

# 함수: 파이썬 환경 확인
check_python_env() {
    log_info "파이썬 환경 확인 중..."
    
    # uv가 설치되어 있는지 확인
    if ! command -v uv &> /dev/null; then
        log_error "uv가 설치되어 있지 않습니다."
        log_info "uv를 설치하거나 Python 가상환경을 활성화하세요."
        exit 1
    fi
    
    # uv를 통해 필요한 패키지 확인
    uv run python -c "import polars, simhash, google.genai" 2>/dev/null || {
        log_error "필요한 Python 패키지가 설치되어 있지 않습니다."
        log_info "다음 명령어로 설치하세요: uv sync"
        exit 1
    }
    
    log_success "파이썬 환경 확인 완료"
}

# 함수: 파일 존재 확인
check_input_file() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        log_error "입력 파일을 찾을 수 없습니다: $input_file"
        exit 1
    fi
    
    local file_size=$(du -h "$input_file" | cut -f1)
    log_info "입력 파일: $input_file (크기: $file_size)"
}

# 함수: 파이프라인 실행
run_pipeline() {
    local input_file="$1"
    local text_col="$2"
    local output_dir="$3"
    local mode="$4"
    local dry_run="$5"
    local skip_flags="$6"
    local sample_size="$7"
    local sample_seed="$8"
    
    # Python 명령어 구성
    local cmd="uv run python -m src.data_pipeline"
    cmd="$cmd --input \"$input_file\""
    cmd="$cmd --text_col \"$text_col\""
    cmd="$cmd --output_dir \"$output_dir\""
    
    # 샘플링 매개변수 추가 (값이 있는 경우에만)
    if [[ -n "$sample_size" ]]; then
        cmd="$cmd --sample_size $sample_size"
    fi
    if [[ -n "$sample_seed" ]]; then
        cmd="$cmd --sample_seed $sample_seed"
    fi
    
    case "$mode" in
        "dedup")
            cmd="$cmd --skip_filtering --skip_aug"
            log_info "중복제거만 실행합니다..."
            ;;
        "filter")
            cmd="$cmd --skip_dedup --skip_aug"
            log_info "데이터 필터링만 실행합니다..."
            ;;
        "aug")
            cmd="$cmd --skip_dedup --skip_filtering"
            log_info "데이터 증강만 실행합니다..."
            ;;
        "status")
            cmd="$cmd --status"
            log_info "파이프라인 상태를 확인합니다..."
            ;;
        "full")
            log_info "전체 파이프라인을 실행합니다..."
            ;;
        "custom")
            cmd="$cmd $skip_flags"
            log_info "사용자 정의 파이프라인을 실행합니다..."
            ;;
    esac
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN - 실행될 명령어:"
        echo "  $cmd"
        return 0
    fi
    
    log_info "실행 명령어: $cmd"
    log_info "파이프라인 시작..."
    
    # 실행 시간 측정
    local start_time=$(date +%s)
    
    # 명령어 실행
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "파이프라인이 성공적으로 완료되었습니다! (소요시간: ${duration}초)"
    else
        log_error "파이프라인 실행 중 오류가 발생했습니다."
        exit 1
    fi
}

# 메인 함수
main() {
    local input_file="$DEFAULT_INPUT_FILE"
    local text_col="$DEFAULT_TEXT_COL"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local mode="full"
    local dry_run="false"
    local skip_flags=""
    local sample_size=""
    local sample_seed=""
    
    # 인자 파싱
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --dedup-only)
                mode="dedup"
                shift
                ;;
            --filter-only)
                mode="filter"
                shift
                ;;
            --aug-only)
                mode="aug"
                shift
                ;;
            --skip-dedup)
                skip_flags="$skip_flags --skip_dedup"
                mode="custom"
                shift
                ;;
            --skip-filter)
                skip_flags="$skip_flags --skip_filtering"
                mode="custom"
                shift
                ;;
            --skip-aug)
                skip_flags="$skip_flags --skip_aug"
                mode="custom"
                shift
                ;;
            --status)
                mode="status"
                shift
                ;;
            --text-col)
                text_col="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --sample-size)
                sample_size="$2"
                shift 2
                ;;
            --sample-seed)
                sample_seed="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            -*)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
            *)
                # 첫 번째 위치 인자는 입력 파일로 처리
                if [[ "$input_file" == "$DEFAULT_INPUT_FILE" ]]; then
                    input_file="$1"
                else
                    log_error "너무 많은 인자입니다: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 환경 및 파일 확인 (status 모드가 아닌 경우에만)
    if [[ "$mode" != "status" ]]; then
        check_python_env
        check_input_file "$input_file"
    fi
    
    # 파이프라인 실행
    run_pipeline "$input_file" "$text_col" "$output_dir" "$mode" "$dry_run" "$skip_flags" "$sample_size" "$sample_seed"
}

# 스크립트 시작
echo "=================================================="
echo "         데이터 처리 파이프라인 자동화"
echo "=================================================="
echo ""

# 메인 함수 실행
main "$@"