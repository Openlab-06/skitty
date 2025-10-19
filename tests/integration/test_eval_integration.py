import pytest
from unittest.mock import Mock, patch, AsyncMock
from eval.evaluation import LLMEvaluation
import asyncio
import json

class TestLLMEvaluationIntegration:
    """LLMEvaluation 통합 테스트 (API 호출 모킹)"""
    
    @pytest.fixture
    def mock_evaluator(self):
        """모킹된 LLMEvaluation 인스턴스"""
        with patch('eval.evaluation.load_dataset') as mock_load_dataset, \
             patch('eval.evaluation.get_config') as mock_config, \
             patch('eval.evaluation.AsyncOpenAI') as mock_openai, \
             patch('eval.evaluation.CrossEncoder') as mock_cross_encoder, \
             patch('eval.evaluation.SentenceBLEU') as mock_bleu:
            
            # 데이터셋 모킹
            mock_dataset = [
                {
                    "instruction": "스팸 메시지를 분류하세요.",
                    "input": "무료 상품!",
                    "output": "스팸입니다."
                },
                {
                    "instruction": "감정을 분석하세요.",
                    "input": "좋은 하루네요.",
                    "output": "긍정적입니다."
                }
            ]
            mock_load_dataset.return_value = mock_dataset
            
            # 환경 설정 모킹
            mock_env = Mock()
            mock_env.SPAM_MODEL_URL = "http://test.com"
            mock_env.SPAM_MODEL_API_KEY = "test-key"
            mock_env.OPENAI_API_KEY = "openai-test-key"
            mock_config.return_value = mock_env
            
            # OpenAI 클라이언트 모킹
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            evaluator = LLMEvaluation()
            evaluator.dataset = mock_dataset
            
            return evaluator, mock_client
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_evaluator):
        """성공적인 평가 실행 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 모델 API 응답 모킹
        model_response = Mock()
        model_response.choices = [Mock()]
        model_response.choices[0].message.content = "예측 결과"
        
        # Judge API 응답 모킹
        judge_response = Mock()
        judge_response.choices = [Mock()]
        judge_response.choices[0].message.content = json.dumps({
            "score": 0.9,
            "explanation": "좋은 예측"
        })
        
        # API 호출 순서: model, judge, model, judge
        mock_client.chat.completions.create.side_effect = [
            model_response,
            judge_response,
            model_response,
            judge_response,
        ]
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.8), \
             patch.object(evaluator, '_calculate_semantic_similarity', return_value=0.75):
            
            results = await evaluator.evaluate()
        
        # 검증
        assert 'bleu' in results
        assert 'semantic_similarity' in results
        assert 'llm_judge' in results
        assert 'metrics_details' in results
        assert 'total_samples' in results
        
        assert results['bleu'] == 0.8
        assert results['semantic_similarity'] == 0.75
        assert results['llm_judge'] == 0.9
        assert results['total_samples'] == 2
        
        # API 호출 횟수 확인 (각 샘플당 model + judge = 4번)
        assert mock_client.chat.completions.create.call_count == 4
    
    @pytest.mark.asyncio
    async def test_evaluate_with_api_errors(self, mock_evaluator):
        """API 오류가 있는 경우 평가 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 모델 응답
        model_response = Mock()
        model_response.choices = [Mock()]
        model_response.choices[0].message.content = "예측 결과"
        
        # Judge 응답
        judge_response = Mock()
        judge_response.choices = [Mock()]
        judge_response.choices[0].message.content = json.dumps({
            "score": 0.5,
            "explanation": "평가 완료"
        })
        
        # 첫 번째 샘플 실패, 두 번째 성공
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),  # 첫 번째 샘플 - 모델 호출 실패
            model_response,          # 두 번째 샘플 - 모델 호출 성공
            judge_response,          # 두 번째 샘플 - judge 호출 성공
        ]
        
        # 평가 메트릭 모킹 (성공한 경우에만)
        with patch.object(evaluator.bleu, 'score', return_value=0.5), \
             patch.object(evaluator, '_calculate_semantic_similarity', return_value=0.4), \
             patch('eval.evaluation.logger') as mock_logger:
            
            results = await evaluator.evaluate()
        
        # 검증
        assert 'bleu' in results
        assert 'semantic_similarity' in results
        assert 'llm_judge' in results
        assert results['bleu'] == 0.5  # 성공한 하나의 샘플만
        assert results['total_samples'] == 1
        
        # 경고 로그가 기록되었는지 확인
        mock_logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_evaluate_all_samples_fail(self, mock_evaluator):
        """모든 샘플이 실패하는 경우 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 모든 API 호출 실패
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('eval.evaluation.logger'):
            with pytest.raises(RuntimeError, match="유효한 샘플이 없어 점수를 계산할 수 없습니다"):
                await evaluator.evaluate()
    
    @pytest.mark.asyncio
    async def test_evaluate_with_empty_prediction(self, mock_evaluator):
        """빈 예측 결과 처리 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 빈 응답 모킹
        model_response = Mock()
        model_response.choices = [Mock()]
        model_response.choices[0].message.content = "   "  # 공백만 있는 응답
        
        judge_response = Mock()
        judge_response.choices = [Mock()]
        judge_response.choices[0].message.content = json.dumps({
            "score": 0.0,
            "explanation": "빈 응답"
        })
        
        mock_client.chat.completions.create.side_effect = [
            model_response, judge_response,
            model_response, judge_response
        ]
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.0), \
             patch.object(evaluator, '_calculate_semantic_similarity', return_value=0.0):
            
            results = await evaluator.evaluate()
        
        # 검증 - 빈 예측도 처리되어야 함
        assert 'bleu' in results
        assert 'semantic_similarity' in results
        assert 'llm_judge' in results
    
    def test_module_import_success(self):
        """모듈 import 성공 테스트 (evaluation.py에는 main이 __name__ == '__main__' 블록에 있음)"""
        # evaluation.py에는 main 함수가 __name__ == '__main__' 블록 안에 있으므로
        # 직접 import할 수 없음. 대신 모듈 import가 성공하는지만 확인
        try:
            import eval.evaluation
            assert hasattr(eval.evaluation, 'LLMEvaluation')
            assert hasattr(eval.evaluation, 'safe_mean')
        except ImportError:
            pytest.fail("eval.evaluation 모듈 import 실패")
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_flow(self, mock_evaluator):
        """전체 평가 플로우 end-to-end 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 실제 evaluation.py의 main 함수 로직을 시뮬레이션
        # 1. 평가 인스턴스 생성은 이미 fixture에서 완료
        
        # 2. API 응답 모킹
        model_response = Mock()
        model_response.choices = [Mock()]
        model_response.choices[0].message.content = "스팸 메시지입니다."
        
        judge_response = Mock()
        judge_response.choices = [Mock()]
        judge_response.choices[0].message.content = json.dumps({
            "score": 0.85,
            "explanation": "정확한 설명입니다."
        })
        
        mock_client.chat.completions.create.side_effect = [
            model_response, judge_response,
            model_response, judge_response
        ]
        
        # 3. 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.75), \
             patch.object(evaluator, '_calculate_semantic_similarity', return_value=0.8), \
             patch('eval.evaluation.logger') as mock_logger:
            
            # 4. 평가 실행 (main 함수의 핵심 로직)
            results = await evaluator.evaluate()
            
            # 5. 결과 검증
            assert 'bleu' in results
            assert 'semantic_similarity' in results
            assert 'llm_judge' in results
            assert 'metrics_details' in results
            
            assert results['bleu'] == 0.75
            assert results['semantic_similarity'] == 0.8
            assert results['llm_judge'] == 0.85
            
            # metrics_details 검증
            assert len(results['metrics_details']) == 2
            for metric in results['metrics_details']:
                assert hasattr(metric, 'sample_id')
                assert hasattr(metric, 'bleu_score')
                assert hasattr(metric, 'semantic_similarity')
                assert hasattr(metric, 'llm_judge_score')
                assert hasattr(metric, 'llm_judge_explanation')
    
    @pytest.mark.asyncio
    async def test_evaluate_temperature_setting(self, mock_evaluator):
        """temperature 설정 확인 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # API 응답 모킹
        model_response = Mock()
        model_response.choices = [Mock()]
        model_response.choices[0].message.content = "예측 결과"
        
        judge_response = Mock()
        judge_response.choices = [Mock()]
        judge_response.choices[0].message.content = json.dumps({
            "score": 0.8,
            "explanation": "평가 완료"
        })
        
        mock_client.chat.completions.create.side_effect = [
            model_response, judge_response,
            model_response, judge_response
        ]
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.8), \
             patch.object(evaluator, '_calculate_semantic_similarity', return_value=0.7):
            
            await evaluator.evaluate()
        
        # temperature 설정 확인
        call_args_list = mock_client.chat.completions.create.call_args_list
        
        # 모델 호출: temperature=0.0
        assert call_args_list[0][1]['temperature'] == 0.0
        assert call_args_list[0][1]['model'] == evaluator.model_name
        
        # Judge 호출: temperature=0.1
        assert call_args_list[1][1]['temperature'] == 0.1
        assert call_args_list[1][1]['model'] == evaluator.judge_model_name