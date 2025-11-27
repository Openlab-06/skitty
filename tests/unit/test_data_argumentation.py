"""data_augmentation.py 모듈에 대한 단위 테스트"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import asyncio
from pathlib import Path

from src.data.data_augmentation import DataAugmentation
from src.data import constants
from src.utils.exception import DataAugmentationError


class TestDataAugmentationInit:
    """DataAugmentation 초기화 테스트"""

    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    def test_init_success(self, mock_get_config, mock_read_csv):
        """정상적인 초기화 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({
            "CN": ["spam text 1", "spam text 2", "spam text 3"],
            "complexity": ["HIGH", "MEDIUM", "LOW"]
        })
        mock_read_csv.return_value = mock_df

        # 테스트 실행
        data_aug = DataAugmentation("test.csv", batch_size=5)

        # 검증
        mock_read_csv.assert_called_once_with("test.csv")
        assert data_aug.batch_size == 5
        assert len(data_aug.data) == 3
        assert "CN" in data_aug.data.columns

    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    def test_init_with_defaults(self, mock_get_config, mock_read_csv):
        """기본값으로 초기화 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({
            "CN": ["spam text 1", "spam text 2"],
            "complexity": ["HIGH", "MEDIUM"]
        })
        mock_read_csv.return_value = mock_df

        # 테스트 실행
        data_aug = DataAugmentation("test.csv")

        # 기본값 검증
        assert data_aug.batch_size == constants.DEFAULT_AUG_BATCH_SIZE

    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    def test_prompt_content(self, mock_get_config, mock_read_csv):
        """프롬프트 내용 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test"], "complexity": ["HIGH"]})
        mock_read_csv.return_value = mock_df

        # 테스트 실행
        data_aug = DataAugmentation("test.csv")

        # 프롬프트 내용 검증
        assert "스팸 문자로 판정한 근거를 생성하는" in data_aug.prompt
        assert "{text}" in data_aug.prompt
        assert "개인 정보 요구" in data_aug.prompt
        assert "심리적 압박" in data_aug.prompt
        assert "링크/URL" in data_aug.prompt
        assert "100자 이상으로" in data_aug.prompt


class TestProcessSingleText:
    """_process_single_text 메서드 테스트"""

    @pytest.fixture
    def data_aug(self):
        """테스트용 DataAugmentation 인스턴스"""
        with patch('src.data.data_augmentation.pd.read_csv') as mock_read_csv, \
             patch('src.data.data_augmentation.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_config.GEMINI_MODEL_ARGU = "gemini-pro"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
            mock_read_csv.return_value = mock_df

            return DataAugmentation("test.csv")

    @pytest.mark.asyncio
    async def test_process_single_text_gemini_success(self, data_aug):
        """Gemini API 성공 시 테스트"""
        # Mock Gemini client
        mock_response = Mock()
        mock_response.text = "이 문자는 개인정보를 요구하고 있어 스팸으로 분류됩니다. 특히 신분증 번호와 비밀번호를 요구하는 내용이 포함되어 있으며, 긴급성을 강조하여 즉각적인 행동을 유도하고 있습니다."
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # 테스트 실행
        result = await data_aug._process_single_text("수상한 스팸 텍스트", 0)

        # 검증
        assert "개인정보를 요구" in result
        assert len(result) > 50  # 충분히 긴 설명
        data_aug.gemini_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_text_gemini_failure_openai_success(self, data_aug):
        """Gemini 실패 후 OpenAI 성공 테스트"""
        # Mock Gemini client - 실패
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Gemini API error")
        )

        # Mock OpenAI client - 성공
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "이 문자는 투자 권유 내용이 포함되어 있고, 단축 URL을 통해 의심스러운 링크로 유도하고 있어 스팸으로 판정됩니다. 또한 기간 한정이라는 심리적 압박을 가하고 있습니다."
        data_aug.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        # 테스트 실행
        result = await data_aug._process_single_text("투자 권유 스팸 텍스트", 0)

        # 검증
        assert "투자 권유" in result
        assert "스팸으로 판정" in result
        data_aug.openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_text_both_apis_fail(self, data_aug):
        """Gemini와 OpenAI 모두 실패 시 테스트"""
        # Mock 둘 다 실패
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Gemini API error")
        )
        data_aug.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )

        # 테스트 실행 및 예외 확인
        with pytest.raises(DataAugmentationError):
            await data_aug._process_single_text("test spam text", 0)


class TestProcessBatch:
    """_process_batch 메서드 테스트"""

    @pytest.fixture
    def data_aug(self):
        """테스트용 DataAugmentation 인스턴스"""
        with patch('src.data.data_augmentation.pd.read_csv') as mock_read_csv, \
             patch('src.data.data_augmentation.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
            mock_read_csv.return_value = mock_df

            return DataAugmentation("test.csv")

    @pytest.mark.asyncio
    async def test_process_batch_success(self, data_aug):
        """배치 처리 성공 테스트"""
        # Mock _process_single_text
        data_aug._process_single_text = AsyncMock(side_effect=[
            "첫 번째 스팸 설명입니다. 개인정보 요구로 인해 스팸으로 분류됩니다.",
            "두 번째 스팸 설명입니다. 투자 권유 내용이 포함되어 있습니다.",
            "세 번째 스팸 설명입니다. 의심스러운 링크가 포함되어 있습니다."
        ])

        batch_data = [(0, "text1"), (1, "text2"), (2, "text3")]

        # 테스트 실행
        results = await data_aug._process_batch(batch_data)

        # 검증
        assert len(results) == 3
        assert "개인정보 요구" in results[0][1]
        assert "투자 권유" in results[1][1]
        assert "의심스러운 링크" in results[2][1]

    @pytest.mark.asyncio
    async def test_process_batch_with_exceptions(self, data_aug):
        """배치 처리 중 일부 실패 테스트"""
        # Mock _process_single_text - 일부 실패
        data_aug._process_single_text = AsyncMock(side_effect=[
            "첫 번째 스팸 설명입니다.",
            Exception("API error"),  # 두 번째 실패
            "세 번째 스팸 설명입니다."
        ])

        batch_data = [(0, "text1"), (1, "text2"), (2, "text3")]

        # 테스트 실행
        results = await data_aug._process_batch(batch_data)

        # 검증
        assert len(results) == 3
        assert results[0][1] == "첫 번째 스팸 설명입니다."
        assert "스팸으로 분류되었습니다" in results[1][1]  # 기본값
        assert results[2][1] == "세 번째 스팸 설명입니다."


class TestDataAugmentation:
    """data_argumentation 메서드 테스트"""

    @pytest.fixture
    def data_aug(self):
        """테스트용 DataAugmentation 인스턴스"""
        with patch('src.data.data_augmentation.pd.read_csv') as mock_read_csv, \
             patch('src.data.data_augmentation.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({
                "CN": ["spam text 1", "spam text 2", "spam text 3", "spam text 4", "spam text 5"],
                "complexity": ["HIGH", "MEDIUM", "LOW", "VERY_HIGH", "EXTREMELY_HIGH"]
            })
            mock_read_csv.return_value = mock_df

            return DataAugmentation("test.csv", batch_size=2)

    @pytest.mark.asyncio
    @patch('src.data.data_augmentation.logger')
    async def test_data_argumentation_success(self, mock_logger, data_aug):
        """데이터 증강 성공 테스트"""
        # Mock _process_batch
        data_aug._process_batch = AsyncMock(side_effect=[
            [(0, "첫 번째 스팸 설명"), (1, "두 번째 스팸 설명")],      # 첫 번째 배치
            [(2, "세 번째 스팸 설명"), (3, "네 번째 스팸 설명")],      # 두 번째 배치
            [(4, "다섯 번째 스팸 설명")]                              # 마지막 배치
        ])

        # Mock DataFrame.to_csv
        data_aug.data.to_csv = Mock()

        # 테스트 실행
        await data_aug.data_argumentation()

        # 검증
        assert data_aug._process_batch.call_count == 3  # 5개 데이터를 배치 크기 2로 처리
        data_aug.data.to_csv.assert_called_once_with("./src/data/final_spam.csv", index=False)
        
        # output 컬럼이 추가되었는지 확인
        assert "output" in data_aug.data.columns

    @pytest.mark.asyncio
    @patch('src.data.data_augmentation.logger')
    async def test_data_argumentation_with_batch_fallback(self, mock_logger, data_aug):
        """배치 실패 시 개별 처리 fallback 테스트"""
        # Mock _process_batch - 첫 번째 배치 실패
        data_aug._process_batch = AsyncMock(side_effect=[
            Exception("Batch processing failed"),
            [(2, "세 번째 스팸 설명"), (3, "네 번째 스팸 설명")],
            [(4, "다섯 번째 스팸 설명")]
        ])
        
        # Mock _process_single_text for fallback
        data_aug._process_single_text = AsyncMock(side_effect=[
            "첫 번째 개별 처리 설명",
            "두 번째 개별 처리 설명"
        ])

        # Mock DataFrame.to_csv
        data_aug.data.to_csv = Mock()

        # 테스트 실행
        await data_aug.data_argumentation()

        # 검증 - fallback이 호출되었는지 확인
        assert data_aug._process_single_text.call_count == 2  # 첫 번째 배치의 2개 아이템

    @pytest.mark.asyncio
    async def test_data_argumentation_individual_fallback_with_failure(self, data_aug):
        """개별 처리 fallback에서도 실패하는 경우 테스트"""
        # Mock _process_batch - 실패
        data_aug._process_batch = AsyncMock(side_effect=Exception("Batch processing failed"))
        
        # Mock _process_single_text - 실패
        data_aug._process_single_text = AsyncMock(side_effect=Exception("Individual processing failed"))

        # Mock DataFrame.to_csv
        data_aug.data.to_csv = Mock()

        # 테스트 실행
        await data_aug.data_argumentation()

        # 검증 - 기본값이 사용되었는지 확인
        output_values = data_aug.data["output"].tolist()
        for val in output_values:
            assert "스팸으로 분류되었습니다" in val


class TestArgumentationIntegration:
    """통합 테스트"""

    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    def test_prompt_formatting(self, mock_get_config, mock_read_csv):
        """프롬프트 포맷팅 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
        mock_read_csv.return_value = mock_df

        # 테스트 실행
        data_aug = DataAugmentation("test.csv")
        
        # 프롬프트 포맷팅 테스트
        formatted_prompt = data_aug.prompt.format(text="테스트 스팸 문자")
        assert "테스트 스팸 문자" in formatted_prompt
        assert "스팸 문자로 판정한 근거" in formatted_prompt

    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    def test_batch_size_configuration(self, mock_get_config, mock_read_csv):
        """배치 크기 설정 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
        mock_read_csv.return_value = mock_df

        # 다양한 배치 크기로 테스트
        for batch_size in [1, 3, 5, 10]:
            data_aug = DataAugmentation("test.csv", batch_size=batch_size)
            assert data_aug.batch_size == batch_size

    @pytest.mark.asyncio
    @patch('src.data.data_augmentation.pd.read_csv')
    @patch('src.data.data_augmentation.get_config')
    async def test_gemini_config_usage(self, mock_get_config, mock_read_csv):
        """Gemini 설정 사용 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_config.GEMINI_MODEL_ARGU = "gemini-2.5-flash"
        mock_config.OPENAI_MODEL = "gpt-4"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
        mock_read_csv.return_value = mock_df

        data_aug = DataAugmentation("test.csv")

        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "테스트 응답"
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # 테스트 실행
        result = await data_aug._process_single_text("test text", 0)

        # 검증
        assert result == "테스트 응답"
        
        # Gemini API 호출 시 올바른 설정이 사용되었는지 확인
        call_args = data_aug.gemini_client.aio.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-flash"  # 실제 사용되는 모델명


class TestMainFunction:
    """__main__ 블록 테스트"""

    @patch('src.data.data_augmentation.asyncio.run')
    @patch('src.data.data_augmentation.DataAugmentation')
    @patch('src.data.data_augmentation.logger')
    def test_main_execution(self, mock_logger, mock_data_argumentation_class, mock_asyncio_run):
        """메인 함수 실행 테스트"""
        # Mock 설정
        mock_instance = Mock()
        mock_data_argumentation_class.return_value = mock_instance

        # __main__ 블록 시뮬레이션
        from src.data.data_augmentation import DataAugmentation
        
        # 직접 DataAugmentation을 생성하고 실행하는 것을 시뮬레이션
        data_aug = DataAugmentation("./src/data/filter_spam_high.csv")
        
        # 인스턴스가 생성되었는지 확인
        assert data_aug is not None


class TestErrorHandling:
    """에러 처리 테스트"""

    @pytest.fixture
    def data_aug(self):
        """테스트용 DataAugmentation 인스턴스"""
        with patch('src.data.data_augmentation.pd.read_csv') as mock_read_csv, \
             patch('src.data.data_augmentation.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({"CN": ["test text"], "complexity": ["HIGH"]})
            mock_read_csv.return_value = mock_df

            return DataAugmentation("test.csv")

    @pytest.mark.asyncio
    async def test_gemini_timeout_error(self, data_aug):
        """Gemini API 타임아웃 에러 테스트"""
        # Mock Gemini client - 타임아웃
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )

        # Mock OpenAI client - 성공
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "OpenAI로 처리된 결과"
        data_aug.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        # 테스트 실행
        result = await data_aug._process_single_text("test text", 0)

        # 검증 - OpenAI로 fallback 되었는지 확인
        assert result == "OpenAI로 처리된 결과"

    @pytest.mark.asyncio
    async def test_network_error_handling(self, data_aug):
        """네트워크 에러 처리 테스트"""
        # Mock 둘 다 네트워크 에러
        data_aug.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Network error")
        )
        data_aug.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # 테스트 실행 및 예외 확인
        with pytest.raises(DataAugmentationError, match="Connection failed"):
            await data_aug._process_single_text("test text", 0)
