import pytest
from unittest.mock import Mock, patch, AsyncMock
from eval.evaluation import LLMEvaluation
import asyncio

class TestLLMEvaluationIntegration:
    """LLMEvaluation 통합 테스트 (API 호출 모킹)"""
    
    @pytest.fixture
    def mock_evaluator(self):
        """모킹된 LLMEvaluation 인스턴스"""
        with patch('eval.evaluation.load_dataset') as mock_load_dataset, \
             patch('eval.evaluation.get_config') as mock_config, \
             patch('eval.evaluation.AsyncOpenAI') as mock_openai:
            
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
            mock_env.SPAM_MODEL = "test-model"
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
        
        # API 응답 모킹
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "예측 결과"
        mock_client.chat.completions.create.return_value = mock_response
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.8), \
             patch.object(evaluator.rouge, 'score', return_value={'rouge1': 0.7, 'rouge2': 0.6}):
            
            results = await evaluator.evaluate()
        
        # 검증
        assert 'BLEU' in results
        assert 'ROUGE' in results
        assert results['BLEU'] == 0.8
        assert results['ROUGE']['rouge1'] == 0.7
        assert results['ROUGE']['rouge2'] == 0.6
        
        # API 호출 횟수 확인 (데이터셋 크기만큼)
        assert mock_client.chat.completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_with_api_errors(self, mock_evaluator):
        """API 오류가 있는 경우 평가 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # 첫 번째 호출은 실패, 두 번째는 성공
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "예측 결과"
        
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),  # 첫 번째 호출 실패
            mock_response             # 두 번째 호출 성공
        ]
        
        # 평가 메트릭 모킹 (성공한 경우에만)
        with patch.object(evaluator.bleu, 'score', return_value=0.5), \
             patch.object(evaluator.rouge, 'score', return_value={'rouge1': 0.4}), \
             patch('eval.evaluation.logger') as mock_logger:
            
            results = await evaluator.evaluate()
        
        # 검증
        assert 'BLEU' in results
        assert 'ROUGE' in results
        assert results['BLEU'] == 0.5  # 성공한 하나의 샘플만
        
        # 경고 로그가 기록되었는지 확인
        mock_logger.warning.assert_called_once()
    
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
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "   "  # 공백만 있는 응답
        mock_client.chat.completions.create.return_value = mock_response
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.0), \
             patch.object(evaluator.rouge, 'score', return_value={'rouge1': 0.0}):
            
            results = await evaluator.evaluate()
        
        # 검증 - 빈 예측도 처리되어야 함
        assert 'BLEU' in results
        assert 'ROUGE' in results
    
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
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "스팸 메시지입니다."
        mock_client.chat.completions.create.return_value = mock_response
        
        # 3. 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.75), \
             patch.object(evaluator.rouge, 'score', return_value={'rouge1': 0.8, 'rouge2': 0.7}), \
             patch('eval.evaluation.logger') as mock_logger:
            
            # 4. 평가 실행 (main 함수의 핵심 로직)
            results = await evaluator.evaluate()
            
            # 5. 결과 검증 (main 함수에서 로그로 출력하는 부분)
            assert 'BLEU' in results
            assert 'ROUGE' in results
            assert results['BLEU'] == 0.75
            assert 'rouge1' in results['ROUGE']
            assert results['ROUGE']['rouge1'] == 0.8
            
            # 로그가 기록되었는지 확인하지 않음 (실제 main에서만 로그 출력)
    
    @pytest.mark.asyncio
    async def test_evaluate_temperature_setting(self, mock_evaluator):
        """temperature=0.0 설정 확인 테스트"""
        evaluator, mock_client = mock_evaluator
        
        # API 응답 모킹
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "예측 결과"
        mock_client.chat.completions.create.return_value = mock_response
        
        # 평가 메트릭 모킹
        with patch.object(evaluator.bleu, 'score', return_value=0.8), \
             patch.object(evaluator.rouge, 'score', return_value={'rouge1': 0.7}):
            
            await evaluator.evaluate()
        
        # temperature=0.0으로 호출되었는지 확인
        call_args = mock_client.chat.completions.create.call_args_list[0]
        assert call_args[1]['temperature'] == 0.0
        assert call_args[1]['model'] == evaluator.model_name