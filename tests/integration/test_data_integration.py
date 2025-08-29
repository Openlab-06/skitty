import pytest
from unittest.mock import Mock, patch
from src.data_pipeline import DataPipeline

# 통합 테스트
class TestDataPipelineIntegration:
    """통합 테스트"""
    
    @pytest.fixture
    def temp_files(self, tmp_path):
        """임시 파일 생성 픽스처"""
        input_file = tmp_path / "input.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # 샘플 CSV 파일 생성
        input_file.write_text("text,label\nhello world,0\nspam message,1\n")
        
        return {
            'input_file': str(input_file),
            'output_dir': str(output_dir)
        }

    @patch('src.data_pipeline.DataArgumentation')
    @patch('src.data_pipeline.DataFiltering')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('asyncio.run')
    def test_integration_pipeline_flow(self, mock_asyncio_run, mock_dedup, mock_filtering_class, mock_aug_class, temp_files):
        """실제 파일 시스템을 사용한 통합 테스트"""
        # Mock 설정
        mock_unique_df = Mock()
        mock_unique_df.height = 2
        mock_dedup.return_value = (mock_unique_df, Mock())
        
        mock_filtering_instance = Mock()
        mock_filtering_class.return_value = mock_filtering_instance
        
        mock_aug_instance = Mock()
        mock_aug_class.return_value = mock_aug_instance
        
        # 파이프라인 실행
        pipeline = DataPipeline(temp_files['input_file'], temp_files['output_dir'])
        
        with patch('src.data_pipeline.logger'):
            pipeline.run_full_pipeline()
        
        # 모든 단계가 실행되었는지 확인
        mock_dedup.assert_called_once()
        # asyncio.run이 두 번 호출되었는지 확인 (filtering + argumentation)
        assert mock_asyncio_run.call_count == 2
        
        # 배치 크기와 함께 올바르게 인스턴스가 생성되었는지 확인
        mock_filtering_class.assert_called_once()
        mock_aug_class.assert_called_once()