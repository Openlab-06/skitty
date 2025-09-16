# pytest 실행 스크립트
# 커버리지와 함께 테스트 실행

echo "🧪 테스트 시작..."

# uv를 통해 pytest 실행 (커버리지 포함)
uv run pytest tests/ \
    --verbose \
    --tb=short \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --durations=10

echo ""
echo "📊 테스트 완료!"
echo "📈 HTML 커버리지 리포트: htmlcov/index.html"