#!/bin/bash

# =============================================================================
# LLM 자동 평가 플랫폼 - Skitty Edition 🐱
# =============================================================================

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
PINK='\033[1;35m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 설정
LIMIT=""
VERBOSE="false"
MODEL_NAME="google/gemma-3-4b-it"

# Skitty 시작 화면
show_skitty_banner() {
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
    echo "${PURPLE}          🐱 AI 모델 자동 평가 플랫폼 🐱          ${NC}"
    echo ""
}

# 도움말 출력
show_help() {
    show_skitty_banner
    echo "${CYAN}사용법:${NC} $0 [옵션]"
    echo ""
    echo "${YELLOW}옵션:${NC}"
    echo "  -h, --help              이 도움말 표시"
    echo "  -l, --limit N           평가할 샘플 수 제한 (기본값: 전체)"
    echo "  -v, --verbose           상세 로그 출력"
    echo "  --model MODEL_NAME      평가할 모델명 (기본값: google/gemma-3-4b-it)"
    echo ""
    echo "${YELLOW}예시:${NC}"
    echo "  $0                          # 전체 데이터로 평가"
    echo "  $0 --limit 10               # 10개 샘플로 테스트 평가"
    echo "  $0 --limit 100 --verbose    # 100개 샘플로 상세 평가"
    echo ""
    echo "${YELLOW}📊 평가 메트릭:${NC}"
    echo "  • BLEU Score (25%): 어휘 수준의 일치도"
    echo "  • Semantic Similarity (35%): 의미론적 유사성"
    echo "  • LLM Judge Score (40%): GPT-4o의 설명 품질 평가"
    echo ""
}

# 진행률 바 표시
show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "${CYAN}["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "] %d%% (%d/%d)${NC}\r" "$percent" "$current" "$total"
}

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 메인 함수
main() {
    # Skitty 배너 출력
    show_skitty_banner
    
    log_info "🚀 LLM 자동 평가 시작"
    echo ""
    
    # 환경 확인
    log_info "🔍 환경 점검 중..."
    
    # Python 버전 확인
    if command -v python &> /dev/null; then
        python_version=$(python --version 2>&1)
        log_success "🐍 $python_version 확인"
    else
        log_error "Python이 설치되지 않았습니다"
        exit 1
    fi
    
    # 평가 스크립트 존재 확인
    if [[ -f "eval/evaluation.py" ]]; then
        log_success "📄 평가 스크립트 확인: eval/evaluation.py"
    else
        log_error "평가 스크립트를 찾을 수 없습니다: eval/evaluation.py"
        exit 1
    fi
    
    echo ""
    log_info "⚡ 자동 평가 실행 중..."
    echo ""
    
    # 진행률 시뮬레이션 (실제로는 Python 스크립트가 로그를 출력할 것)
    echo -e "${PURPLE}🐱 Skitty가 열심히 평가 중이에요...${NC}"
    echo ""
    
    # 시작 시간 기록
    start_time=$(date +%s)
    
    # Python 스크립트 실행
    if python eval/evaluation.py --limit "$LIMIT" --verbose "$VERBOSE" --model "$MODEL_NAME"; then
        # 종료 시간 계산
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo ""
        echo "${PURPLE}        ∩───∩        ${NC}"
        echo "${PURPLE}       (  ◕   ◕ )      ${NC}" 
        echo "${PURPLE}      /  \\___/  \\     ${NC}"
        echo "${PURPLE}     (    ___    )    ${NC}"
        echo "${PURPLE}      \\___________/     ${NC}"
        echo "${PINK}         ∪     ∪        ${NC}"
        echo ""
        log_success "🎉 평가 완료! Skitty가 성공했어요!"
        log_info "⏱️  총 소요 시간: ${duration}초"
        echo ""
        echo "${PURPLE}🐱 평가 결과를 확인해주세요! 냥~ 🐱${NC}"
        
    else
        echo ""
        echo "${PURPLE}        ∩───∩        ${NC}"
        echo "${PURPLE}       (  ◕   ◕ )      ${NC}" 
        echo "${PURPLE}      /  \\___/  \\     ${NC}"
        echo "${PURPLE}     (    ___    )    ${NC}"
        echo "${PURPLE}      \\___________/     ${NC}"
        echo "${PINK}         ∪     ∪        ${NC}"
        echo ""
        log_error "❌ 평가 실행 중 오류가 발생했습니다"
        echo "${PURPLE}🐱 Skitty가 슬퍼해요... 다시 시도해주세요 😿${NC}"
        exit 1
    fi
}

# 스크립트 종료 시 정리 작업
cleanup() {
    echo ""
    log_info "🧹 정리 작업 완료"
    echo "${PURPLE}🐱 또 만나요! 냥~ 🐱${NC}"
    echo ""
}

# 시그널 핸들러 등록
trap cleanup EXIT

# 메인 함수 실행
main "$@"