"""data_filtering.py 모듈에 대한 단위 테스트"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import asyncio
from pathlib import Path

from src.data.data_filtering import DataFiltering
from src.config.data_config import DataConfig
from src.utils.exception import DataFilteringError


class TestDataFilteringInit:
    """DataFiltering 초기화 테스트"""

    @patch('src.data.data_filtering.pd.read_parquet')
    @patch('src.data.data_filtering.get_config')
    def test_init_success(self, mock_get_config, mock_read_parquet):
        """정상적인 초기화 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({
            "CN": ["text1", "text2", "text3", "text4", "text5"],
            "label": [0, 1, 0, 1, 0]
        })
        mock_read_parquet.return_value = mock_df

        # 테스트 실행
        data_filter = DataFiltering("test.parquet", sample_size=1.0, sample_seed=42, batch_size=3)

        # 검증
        mock_read_parquet.assert_called_once_with("test.parquet")
        assert data_filter.batch_size == 3
        assert len(data_filter.data) == 5  # 5개 모두 사용 (100% 샘플링)
        assert "CN" in data_filter.data.columns

    @patch('src.data.data_filtering.pd.read_parquet')
    @patch('src.data.data_filtering.get_config')
    def test_init_with_defaults(self, mock_get_config, mock_read_parquet):
        """기본값으로 초기화 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({
            "CN": ["text1"] * 20,  # 적당한 크기 데이터셋
            "label": [0] * 20
        })
        mock_read_parquet.return_value = mock_df

        # 테스트 실행
        data_filter = DataFiltering("test.parquet")

        # 기본값 검증
        assert data_filter.batch_size == DataConfig.DEFAULT_FILTER_BATCH_SIZE
        assert len(data_filter.data) == 1  # 20 * 0.05 (기본 샘플 사이즈)

    @patch('src.data.data_filtering.pd.read_parquet')
    @patch('src.data.data_filtering.get_config')
    def test_init_missing_cn_column(self, mock_get_config, mock_read_parquet):
        """CN 컬럼이 없는 경우 에러 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({
            "text": ["text1", "text2"],  # CN 컬럼이 없음
            "label": [0, 1]
        })
        mock_read_parquet.return_value = mock_df

        # 테스트 실행 및 예외 확인
        with pytest.raises(DataFilteringError, match="Input data must contain 'CN' column"):
            DataFiltering("test.parquet")


class TestProcessSingleText:
    """_process_single_text 메서드 테스트"""

    @pytest.fixture
    def data_filter(self):
        """테스트용 DataFiltering 인스턴스"""
        with patch('src.data.data_filtering.pd.read_parquet') as mock_read_parquet, \
             patch('src.data.data_filtering.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_config.GEMINI_MODEL_FILTER = "gemini-pro"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({"CN": ["test text"] * 20, "label": [1] * 20})
            mock_read_parquet.return_value = mock_df

            return DataFiltering("test.parquet", sample_size=1.0)  # 모든 데이터 사용

    @pytest.mark.asyncio
    async def test_process_single_text_gemini_success(self, data_filter):
        """Gemini API 성공 시 테스트"""
        # Mock Gemini client
        mock_response = Mock()
        mock_response.text = "HIGH"
        data_filter.gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # 테스트 실행
        result = await data_filter._process_single_text("test spam text", 0)

        # 검증
        assert result == "HIGH"
        data_filter.gemini_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_text_gemini_empty_response(self, data_filter):
        """Gemini API 빈 응답 처리 테스트"""
        # Mock Gemini client - text 속성이 None
        mock_response = Mock()
        mock_response.text = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "MEDIUM"
        data_filter.gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # 테스트 실행
        result = await data_filter._process_single_text("test spam text", 0)

        # 검증
        assert result == "MEDIUM"

    @pytest.mark.asyncio
    async def test_process_single_text_gemini_failure_openai_success(self, data_filter):
        """Gemini 실패 후 OpenAI 성공 테스트"""
        # Mock Gemini client - 실패
        data_filter.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Gemini API error")
        )

        # Mock OpenAI client - 성공
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "LOW"
        data_filter.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        # 테스트 실행
        result = await data_filter._process_single_text("test spam text", 0)

        # 검증
        assert result == "LOW"
        data_filter.openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_text_both_apis_fail(self, data_filter):
        """Gemini와 OpenAI 모두 실패 시 테스트"""
        # Mock 둘 다 실패
        data_filter.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Gemini API error")
        )
        data_filter.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )

        # 테스트 실행 및 예외 확인
        with pytest.raises(DataFilteringError):
            await data_filter._process_single_text("test spam text", 0)


class TestProcessBatch:
    """_process_batch 메서드 테스트"""

    @pytest.fixture
    def data_filter(self):
        """테스트용 DataFiltering 인스턴스"""
        with patch('src.data.data_filtering.pd.read_parquet') as mock_read_parquet, \
             patch('src.data.data_filtering.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({"CN": ["test text"] * 20, "label": [1] * 20})
            mock_read_parquet.return_value = mock_df

            return DataFiltering("test.parquet", sample_size=1.0)

    @pytest.mark.asyncio
    async def test_process_batch_success(self, data_filter):
        """배치 처리 성공 테스트"""
        # Mock _process_single_text
        data_filter._process_single_text = AsyncMock(side_effect=["HIGH", "LOW", "MEDIUM"])

        batch_data = [(0, "text1"), (1, "text2"), (2, "text3")]

        # 테스트 실행
        results = await data_filter._process_batch(batch_data)

        # 검증
        assert len(results) == 3
        assert results == [(0, "HIGH"), (1, "LOW"), (2, "MEDIUM")]

    @pytest.mark.asyncio
    async def test_process_batch_with_exceptions(self, data_filter):
        """배치 처리 중 일부 실패 테스트"""
        # Mock _process_single_text - 일부 실패
        data_filter._process_single_text = AsyncMock(side_effect=[
            "HIGH", 
            Exception("API error"),  # 두 번째 실패
            "LOW"
        ])

        batch_data = [(0, "text1"), (1, "text2"), (2, "text3")]

        # 테스트 실행
        results = await data_filter._process_batch(batch_data)

        # 검증
        assert len(results) == 3
        assert results[0] == (0, "HIGH")
        assert results[1] == (1, "MEDIUM")  # 기본값
        assert results[2] == (2, "LOW")


class TestDataFilter:
    """data_filter 메서드 테스트"""

    @pytest.fixture
    def data_filter(self):
        """테스트용 DataFiltering 인스턴스"""
        with patch('src.data.data_filtering.pd.read_parquet') as mock_read_parquet, \
             patch('src.data.data_filtering.get_config') as mock_get_config:
            
            mock_config = Mock()
            mock_config.GEMINI_API_KEY = "test_gemini_key"
            mock_config.OPENAI_API_KEY = "test_openai_key"
            mock_get_config.return_value = mock_config

            mock_df = pd.DataFrame({
                "CN": ["spam text 1", "spam text 2", "spam text 3", "spam text 4", "spam text 5"],
                "label": [1, 1, 1, 1, 1]
            })
            mock_read_parquet.return_value = mock_df

            return DataFiltering("test.parquet", sample_size=1.0, batch_size=2)

    @pytest.mark.asyncio
    @patch('src.data.data_filtering.logger')
    async def test_data_filter_success(self, mock_logger, data_filter):
        """데이터 필터링 성공 테스트"""
        # Mock _process_batch
        data_filter._process_batch = AsyncMock(side_effect=[
            [(0, "HIGH"), (1, "LOW")],      # 첫 번째 배치
            [(2, "MEDIUM"), (3, "HIGH")],   # 두 번째 배치
            [(4, "LOW")]                     # 마지막 배치
        ])

        # Mock DataFrame.to_csv
        data_filter.data.to_csv = Mock()

        # 테스트 실행
        await data_filter.data_filter()

        # 검증
        assert data_filter._process_batch.call_count == 3  # 5개 데이터를 배치 크기 2로 처리
        data_filter.data.to_csv.assert_called_once_with("./src/data/filter_spam.csv", index=False)
        
        # complexity 컬럼이 추가되었는지 확인
        assert "complexity" in data_filter.data.columns

    @pytest.mark.asyncio
    @patch('src.data.data_filtering.logger')
    async def test_data_filter_with_batch_fallback(self, mock_logger, data_filter):
        """배치 실패 시 개별 처리 fallback 테스트"""
        # Mock _process_batch - 첫 번째 배치 실패
        data_filter._process_batch = AsyncMock(side_effect=[
            Exception("Batch processing failed"),
            [(2, "MEDIUM"), (3, "HIGH")],
            [(4, "LOW")]
        ])
        
        # Mock _process_single_text for fallback
        data_filter._process_single_text = AsyncMock(side_effect=["HIGH", "LOW"])

        # Mock DataFrame.to_csv
        data_filter.data.to_csv = Mock()

        # 테스트 실행
        await data_filter.data_filter()

        # 검증 - fallback이 호출되었는지 확인
        assert data_filter._process_single_text.call_count == 2  # 첫 번째 배치의 2개 아이템

    @pytest.mark.asyncio
    async def test_data_filter_individual_fallback_with_failure(self, data_filter):
        """개별 처리 fallback에서도 실패하는 경우 테스트"""
        # Mock _process_batch - 실패
        data_filter._process_batch = AsyncMock(side_effect=Exception("Batch processing failed"))
        
        # Mock _process_single_text - 실패
        data_filter._process_single_text = AsyncMock(side_effect=Exception("Individual processing failed"))

        # Mock DataFrame.to_csv
        data_filter.data.to_csv = Mock()

        # 테스트 실행
        await data_filter.data_filter()

        # 검증 - 기본값 "MEDIUM"이 사용되었는지 확인
        complexity_values = data_filter.data["complexity"].tolist()
        assert all(val == "MEDIUM" for val in complexity_values)


class TestDataFilteringIntegration:
    """통합 테스트"""

    @patch('src.data.data_filtering.pd.read_parquet')
    @patch('src.data.data_filtering.get_config')
    def test_prompt_template_formatting(self, mock_get_config, mock_read_parquet):
        """프롬프트 템플릿 포맷팅 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test text"] * 20, "label": [1] * 20})
        mock_read_parquet.return_value = mock_df

        # 테스트 실행
        data_filter = DataFiltering("test.parquet")
        
        # 프롬프트 템플릿이 올바르게 설정되었는지 확인
        assert "복잡도를 기준으로" in data_filter.prompt_template
        assert "{text}" in data_filter.prompt_template
        assert "LOW, MEDIUM, HIGH, VERY_HIGH, EXTREMELY_HIGH" in data_filter.prompt_template

    @patch('src.data.data_filtering.pd.read_parquet')
    @patch('src.data.data_filtering.get_config')
    def test_batch_size_configuration(self, mock_get_config, mock_read_parquet):
        """배치 크기 설정 테스트"""
        # Mock 설정
        mock_config = Mock()
        mock_config.GEMINI_API_KEY = "test_gemini_key"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        mock_get_config.return_value = mock_config

        mock_df = pd.DataFrame({"CN": ["test text"] * 20, "label": [1] * 20})
        mock_read_parquet.return_value = mock_df

        # 다양한 배치 크기로 테스트
        for batch_size in [1, 5, 10, 20]:
            data_filter = DataFiltering("test.parquet", batch_size=batch_size)
            assert data_filter.batch_size == batch_size


class TestMainFunction:
    """__main__ 블록 테스트"""

    @patch('src.data.data_filtering.asyncio.run')
    @patch('src.data.data_filtering.DataFiltering')
    @patch('src.data.data_filtering.logger')
    def test_main_execution(self, mock_logger, mock_data_filtering_class, mock_asyncio_run):
        """메인 함수 실행 테스트"""
        # Mock 설정
        mock_instance = Mock()
        mock_data_filtering_class.return_value = mock_instance

        # __main__ 블록 시뮬레이션
        from src.data.data_filtering import DataFiltering
        
        # 직접 DataFiltering을 생성하고 실행하는 것을 시뮬레이션
        data_filter = DataFiltering("./src/data/deduplicated_result.parquet")
        
        # 인스턴스가 생성되었는지 확인
        assert data_filter is not None
