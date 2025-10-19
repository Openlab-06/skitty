import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
from eval.evaluation import LLMEvaluation, safe_mean, EvaluationMetrics


class TestSafeMean:
    """safe_mean 함수 단위 테스트"""
    
    def test_safe_mean_with_values(self):
        """값이 있는 리스트의 평균 계산 테스트"""
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert safe_mean([0.5, 1.5]) == 1.0
        assert safe_mean([10.0]) == 10.0
    
    def test_safe_mean_with_empty_list(self):
        """빈 리스트일 때 0.0 반환 테스트"""
        assert safe_mean([]) == 0.0
    
    def test_safe_mean_with_zeros(self):
        """0값들의 평균 계산 테스트"""
        assert safe_mean([0.0, 0.0, 0.0]) == 0.0


class TestEvaluationMetrics:
    """EvaluationMetrics 데이터클래스 테스트"""
    
    def test_evaluation_metrics_creation(self):
        """EvaluationMetrics 인스턴스 생성 테스트"""
        metrics = EvaluationMetrics(
            sample_id=0,
            prompt="테스트 프롬프트",
            reference="레퍼런스 텍스트",
            prediction="예측 텍스트",
            bleu_score=0.8,
            semantic_similarity=0.75,
            llm_judge_score=0.9,
            llm_judge_explanation="좋은 예측입니다."
        )
        
        assert metrics.sample_id == 0
        assert metrics.prompt == "테스트 프롬프트"
        assert metrics.reference == "레퍼런스 텍스트"
        assert metrics.prediction == "예측 텍스트"
        assert metrics.bleu_score == 0.8
        assert metrics.semantic_similarity == 0.75
        assert metrics.llm_judge_score == 0.9
        assert metrics.llm_judge_explanation == "좋은 예측입니다."


class TestLLMEvaluationUnit:
    """LLMEvaluation 클래스 단위 테스트"""
    
    @pytest.fixture
    def mock_dataset(self):
        """테스트용 데이터셋 픽스처"""
        return [
            {
                "instruction": "스팸 메시지를 분류하세요.",
                "input": "무료 상품 제공!",
                "output": "이것은 스팸 메시지입니다."
            },
            {
                "instruction": "메시지의 감정을 분석하세요.",
                "input": "",
                "output": "중립적인 메시지입니다."
            }
        ]
    
    @patch('eval.evaluation.load_dataset')
    @patch('eval.evaluation.get_config')
    @patch('eval.evaluation.AsyncOpenAI')
    @patch('eval.evaluation.CrossEncoder')
    @patch('eval.evaluation.SentenceBLEU')
    def test_init_with_limit(self, mock_bleu, mock_cross_encoder, mock_openai, mock_config, mock_load_dataset):
        """limit 파라미터가 있는 생성자 테스트"""
        # Mock 설정
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        mock_env = Mock()
        mock_env.SPAM_MODEL_URL = "http://test.com"
        mock_env.SPAM_MODEL_API_KEY = "test-key"
        mock_env.OPENAI_API_KEY = "openai-test-key"
        mock_config.return_value = mock_env
        
        # 테스트 실행
        evaluator = LLMEvaluation(limit=100, model_name="google/gemma-3-4b-it")
        
        # 검증
        mock_load_dataset.assert_called_once_with("Devocean-06/Spam_QA-Corpus", split="test")
        mock_dataset.select.assert_called_once_with(range(100))
        assert evaluator.model_name == "google/gemma-3-4b-it"
        assert evaluator.judge_model_name == "gpt-4o"
    
    @patch('eval.evaluation.load_dataset')
    @patch('eval.evaluation.get_config')
    @patch('eval.evaluation.AsyncOpenAI')
    @patch('eval.evaluation.CrossEncoder')
    @patch('eval.evaluation.SentenceBLEU')
    def test_init_without_limit(self, mock_bleu, mock_cross_encoder, mock_openai, mock_config, mock_load_dataset):
        """limit 파라미터가 없는 생성자 테스트"""
        # Mock 설정
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        mock_env = Mock()
        mock_env.SPAM_MODEL_URL = "http://test.com"
        mock_env.SPAM_MODEL_API_KEY = "test-key"
        mock_env.OPENAI_API_KEY = "openai-test-key"
        mock_config.return_value = mock_env
        
        # 테스트 실행
        evaluator = LLMEvaluation(limit=None)
        
        # 검증
        mock_load_dataset.assert_called_once_with("Devocean-06/Spam_QA-Corpus", split="test")
        mock_dataset.select.assert_not_called()
    
    def test_build_prompt_with_input(self):
        """input이 있는 경우 프롬프트 생성 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.AsyncOpenAI'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            evaluator = LLMEvaluation()
            
            sample = {
                "instruction": "스팸 메시지를 분류하세요.",
                "input": "무료 상품 제공!",
                "output": "이것은 스팸 메시지입니다."
            }
            
            prompt, reference = evaluator._build_prompt(sample)
            
            expected_prompt = "스팸 메시지를 분류하세요.\n\nInput: 무료 상품 제공!"
            assert prompt == expected_prompt
            assert reference == "이것은 스팸 메시지입니다."
    
    def test_build_prompt_without_input(self):
        """input이 없는 경우 프롬프트 생성 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.AsyncOpenAI'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            evaluator = LLMEvaluation()
            
            sample = {
                "instruction": "메시지의 감정을 분석하세요.",
                "input": "",
                "output": "중립적인 메시지입니다."
            }
            
            prompt, reference = evaluator._build_prompt(sample)
            
            assert prompt == "메시지의 감정을 분석하세요."
            assert reference == "중립적인 메시지입니다."
    
    def test_build_prompt_strips_whitespace(self):
        """공백 제거 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.AsyncOpenAI'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            evaluator = LLMEvaluation()
            
            sample = {
                "instruction": "  분류하세요.  ",
                "input": "  테스트 입력  ",
                "output": "  테스트 출력  "
            }
            
            prompt, reference = evaluator._build_prompt(sample)
            
            expected_prompt = "분류하세요.\n\nInput: 테스트 입력"
            assert prompt == expected_prompt
            assert reference == "테스트 출력"
    
    def test_calculate_semantic_similarity(self):
        """의미론적 유사성 계산 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.AsyncOpenAI'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            # CrossEncoder mock 설정
            mock_cross_encoder = Mock()
            mock_cross_encoder.predict.return_value = [0.5]  # -1~1 범위
            
            with patch('eval.evaluation.CrossEncoder', return_value=mock_cross_encoder):
                evaluator = LLMEvaluation()
                
                similarity = evaluator._calculate_semantic_similarity(
                    "이것은 스팸입니다.", 
                    "이것은 스팸 메시지입니다."
                )
                
                # 0.5를 0~1 범위로 정규화: (0.5 + 1) / 2 = 0.75
                assert similarity == 0.75
                mock_cross_encoder.predict.assert_called_once()
    
    def test_calculate_semantic_similarity_error_handling(self):
        """의미론적 유사성 계산 오류 처리 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.AsyncOpenAI'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            # CrossEncoder가 오류를 발생시키도록 설정
            mock_cross_encoder = Mock()
            mock_cross_encoder.predict.side_effect = Exception("Model error")
            
            with patch('eval.evaluation.CrossEncoder', return_value=mock_cross_encoder), \
                 patch('eval.evaluation.logger') as mock_logger:
                
                evaluator = LLMEvaluation()
                
                similarity = evaluator._calculate_semantic_similarity(
                    "텍스트1", 
                    "텍스트2"
                )
                
                # 오류 시 0.0 반환
                assert similarity == 0.0
                mock_logger.warning.assert_called_once()


class TestLLMJudgeEvaluation:
    """LLM Judge 평가 테스트"""
    
    @pytest.mark.asyncio
    async def test_llm_judge_evaluation_success(self):
        """LLM Judge 평가 성공 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            # Judge 모델 mock
            mock_judge = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "score": 0.85,
                "explanation": "좋은 설명입니다."
            })
            mock_judge.chat.completions.create.return_value = mock_response
            
            with patch('eval.evaluation.AsyncOpenAI', return_value=mock_judge):
                evaluator = LLMEvaluation()
                
                score, explanation = await evaluator._llm_judge_evaluation(
                    reference="이것은 스팸입니다.",
                    prediction="이것은 스팸 메시지입니다.",
                    spam_text="무료 상품!"
                )
                
                assert score == 0.85
                assert explanation == "좋은 설명입니다."
    
    @pytest.mark.asyncio
    async def test_llm_judge_evaluation_json_parse_error(self):
        """LLM Judge JSON 파싱 오류 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            # 잘못된 JSON 응답
            mock_judge = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Invalid JSON"
            mock_judge.chat.completions.create.return_value = mock_response
            
            with patch('eval.evaluation.AsyncOpenAI', return_value=mock_judge), \
                 patch('eval.evaluation.logger') as mock_logger:
                
                evaluator = LLMEvaluation()
                
                score, explanation = await evaluator._llm_judge_evaluation(
                    reference="레퍼런스",
                    prediction="예측",
                    spam_text="텍스트"
                )
                
                assert score == 0.0
                assert explanation == "JSON 파싱 실패"
                mock_logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_llm_judge_evaluation_api_error(self):
        """LLM Judge API 오류 테스트"""
        with patch('eval.evaluation.load_dataset'), \
             patch('eval.evaluation.get_config'), \
             patch('eval.evaluation.CrossEncoder'), \
             patch('eval.evaluation.SentenceBLEU'):
            
            # API 오류
            mock_judge = AsyncMock()
            mock_judge.chat.completions.create.side_effect = Exception("API Error")
            
            with patch('eval.evaluation.AsyncOpenAI', return_value=mock_judge), \
                 patch('eval.evaluation.logger') as mock_logger:
                
                evaluator = LLMEvaluation()
                
                score, explanation = await evaluator._llm_judge_evaluation(
                    reference="레퍼런스",
                    prediction="예측",
                    spam_text="텍스트"
                )
                
                assert score == 0.0
                assert "평가 중 오류" in explanation
                mock_logger.warning.assert_called()


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
        
        # 두 개의 다른 응답 (모델, judge)
        mock_client.chat.completions.create.side_effect = [
            model_response,  # 첫 번째 샘플 - 모델 호출
            judge_response,  # 첫 번째 샘플 - judge 호출
            model_response,  # 두 번째 샘플 - 모델 호출
            judge_response,  # 두 번째 샘플 - judge 호출
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
        
        # API 호출 횟수 확인 (각 샘플당 모델 + judge = 2번, 총 4번)
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
        
        # 첫 번째 샘플은 실패, 두 번째는 성공
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
        """모듈 import 성공 테스트"""
        # evaluation.py에는 main 함수가 __name__ == '__main__' 블록 안에 있으므로
        # 직접 import할 수 없음. 대신 모듈 import가 성공하는지만 확인
        try:
            import eval.evaluation
            assert hasattr(eval.evaluation, 'LLMEvaluation')
            assert hasattr(eval.evaluation, 'safe_mean')
        except ImportError:
            pytest.fail("eval.evaluation 모듈 import 실패")
    
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
        
        # temperature 설정 확인 (모델은 0.0, judge는 0.1)
        call_args_list = mock_client.chat.completions.create.call_args_list
        
        # 첫 번째 호출 (모델): temperature=0.0
        assert call_args_list[0][1]['temperature'] == 0.0
        assert call_args_list[0][1]['model'] == evaluator.model_name
        
        # 두 번째 호출 (judge): temperature=0.1
        assert call_args_list[1][1]['temperature'] == 0.1
        assert call_args_list[1][1]['model'] == evaluator.judge_model_name