import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import polars as pl
from src.data_pipeline import DataPipeline


class TestDataPipeline:
    """DataPipeline 클래스 테스트"""

    @pytest.fixture
    def mock_paths(self):
        """테스트용 파일 경로 픽스처"""
        return {
            'input_csv': '/test/input.csv',
            'output_dir': '/test/output',
            'unique_output': Path('/test/output/deduplicated_result.parquet'),
            'dups_output': Path('/test/output/duplicate_analysis.parquet'),
            'final_output': Path('/test/output/final_spam.csv')
        }

    @pytest.fixture
    def pipeline(self, mock_paths):
        """DataPipeline 인스턴스 픽스처"""
        return DataPipeline(mock_paths['input_csv'], mock_paths['output_dir'])

    def test_init(self, mock_paths):
        """__init__ 메서드 테스트"""
        # 실제 Path 객체를 사용하되, exists() 메서드만 모킹
        pipeline = DataPipeline(mock_paths['input_csv'], mock_paths['output_dir'])
        
        assert str(pipeline.input_csv_path) == mock_paths['input_csv']
        assert str(pipeline.output_dir) == mock_paths['output_dir']
        
        # 출력 파일 경로가 올바르게 설정되는지 확인
        assert pipeline.unique_output.name == "deduplicated_result.parquet"
        assert pipeline.dups_output.name == "duplicate_analysis.parquet"
        assert pipeline.final_output.name == "final_spam.csv"

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.DataArgumentation')
    @patch('src.data_pipeline.DataFiltering') 
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    @patch('asyncio.run')  # asyncio.run 모킹 추가
    def test_run_full_pipeline_all_steps(self, mock_asyncio_run, mock_logger, mock_dedup, mock_filtering_class, mock_aug_class, mock_exists, pipeline):
        """모든 단계가 실행되는 경우 테스트"""
        # Mock 설정
        mock_exists.return_value = True  # 모든 파일이 존재한다고 가정
        
        mock_unique_df = Mock()
        mock_unique_df.height = 1000
        mock_dup_df = Mock()
        mock_dedup.return_value = (mock_unique_df, mock_dup_df)
        
        mock_filtering_instance = Mock()
        mock_filtering_class.return_value = mock_filtering_instance
        
        mock_aug_instance = Mock()
        mock_aug_class.return_value = mock_aug_instance
        
        # 테스트 실행
        pipeline.run_full_pipeline()
        
        # 검증
        mock_dedup.assert_called_once()
        # asyncio.run이 두 번 호출되었는지 확인 (filtering + argumentation)
        assert mock_asyncio_run.call_count == 2
        
        # 로그 메시지 확인
        assert mock_logger.info.call_count >= 5

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.DataArgumentation')
    @patch('src.data_pipeline.DataFiltering')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    @patch('asyncio.run')
    def test_run_full_pipeline_skip_dedup(self, mock_asyncio_run, mock_logger, mock_dedup, mock_filtering_class, mock_aug_class, mock_exists, pipeline):
        """중복제거 건너뛰기 테스트"""
        # unique_output과 final_output 파일이 존재한다고 모킹
        mock_exists.return_value = True
        
        mock_filtering_instance = Mock()
        mock_filtering_class.return_value = mock_filtering_instance
        
        mock_aug_instance = Mock()
        mock_aug_class.return_value = mock_aug_instance
        
        # 테스트 실행
        pipeline.run_full_pipeline(run_dedup=False)
        
        # 검증
        mock_dedup.assert_not_called()
        # asyncio.run이 두 번 호출되었는지 확인
        assert mock_asyncio_run.call_count == 2

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    def test_run_full_pipeline_missing_dedup_file(self, mock_logger, mock_dedup, mock_exists, pipeline):
        """중복제거 파일이 없는 경우 에러 테스트"""
        # unique_output 파일이 존재하지 않는다고 모킹
        mock_exists.return_value = False
        
        # 테스트 실행 및 예외 확인
        with pytest.raises(FileNotFoundError):
            pipeline.run_full_pipeline(run_dedup=False)

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.DataArgumentation')
    @patch('src.data_pipeline.DataFiltering')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    @patch('asyncio.run')
    def test_run_full_pipeline_skip_filtering(self, mock_asyncio_run, mock_logger, mock_dedup, mock_filtering_class, mock_aug_class, mock_exists, pipeline):
        """데이터 필터링 건너뛰기 테스트"""
        # Mock 설정
        mock_exists.return_value = True
        
        mock_unique_df = Mock()
        mock_unique_df.height = 1000
        mock_dup_df = Mock()
        mock_dedup.return_value = (mock_unique_df, mock_dup_df)
        
        mock_aug_instance = Mock()
        mock_aug_class.return_value = mock_aug_instance
        
        # 테스트 실행
        pipeline.run_full_pipeline(run_filtering=False)
        
        # 검증
        mock_dedup.assert_called_once()
        mock_filtering_class.assert_not_called()
        # asyncio.run이 한 번만 호출되었는지 확인 (argumentation만)
        assert mock_asyncio_run.call_count == 1
        
        # 증강 단계에서 unique_output을 입력으로 사용했는지 확인
        mock_aug_class.assert_called_with(str(pipeline.unique_output), batch_size=20)

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.DataFiltering')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    @patch('asyncio.run')
    def test_run_full_pipeline_skip_argumentation(self, mock_asyncio_run, mock_logger, mock_dedup, mock_filtering_class, mock_exists, pipeline):
        """데이터 증강 건너뛰기 테스트"""
        # Mock 설정
        mock_exists.return_value = True
        
        mock_unique_df = Mock()
        mock_unique_df.height = 1000
        mock_dup_df = Mock()
        mock_dedup.return_value = (mock_unique_df, mock_dup_df)
        
        mock_filtering_instance = Mock()
        mock_filtering_class.return_value = mock_filtering_instance
        
        # 테스트 실행
        pipeline.run_full_pipeline(run_argumentation=False)
        
        # 검증
        mock_dedup.assert_called_once()
        # asyncio.run이 한 번만 호출되었는지 확인 (filtering만)
        assert mock_asyncio_run.call_count == 1

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.dedup_csv_file')
    @patch('src.data_pipeline.logger')
    def test_run_full_pipeline_exception_handling(self, mock_logger, mock_dedup, mock_exists, pipeline):
        """예외 처리 테스트"""
        # Mock 설정
        mock_exists.return_value = True
        
        # dedup_csv_file에서 예외 발생
        mock_dedup.side_effect = Exception("Test error")
        
        # 테스트 실행 및 예외 확인
        with pytest.raises(Exception, match="Test error"):
            pipeline.run_full_pipeline()
        
        # 에러 로그 확인
        mock_logger.error.assert_called()

    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.exists')
    def test_get_pipeline_status_all_files_exist(self, mock_exists, mock_stat, pipeline):
        """모든 파일이 존재하는 경우 상태 확인 테스트"""
        # Mock 파일 존재 상태
        mock_exists.return_value = True
        
        # Mock 파일 크기
        mock_stat_result = Mock()
        mock_stat_result.st_size = 1024 * 1024  # 1MB
        mock_stat.return_value = mock_stat_result
        
        # 테스트 실행
        status = pipeline.get_pipeline_status()
        
        # 검증
        expected_status = {
            "input_exists": True,
            "deduplicated_exists": True,
            "duplicates_analysis_exists": True,
            "final_output_exists": True,
            "input_size_mb": 1.0,
            "final_size_mb": 1.0
        }
        
        assert status == expected_status

    @patch('pathlib.Path.exists')
    def test_get_pipeline_status_no_files_exist(self, mock_exists, pipeline):
        """파일이 존재하지 않는 경우 상태 확인 테스트"""
        # Mock 파일 존재 상태
        mock_exists.return_value = False
        
        # 테스트 실행
        status = pipeline.get_pipeline_status()
        
        # 검증
        expected_status = {
            "input_exists": False,
            "deduplicated_exists": False,
            "duplicates_analysis_exists": False,
            "final_output_exists": False
        }
        
        assert status == expected_status

    @patch('pathlib.Path.exists')
    @patch('src.data_pipeline.DataArgumentation')
    @patch('asyncio.run')
    def test_argumentation_input_selection(self, mock_asyncio_run, mock_aug_class, mock_exists, pipeline):
        """데이터 증강 단계에서 입력 파일 선택 로직 테스트"""
        # 필터링이 실행되고 final_output이 존재하는 경우
        mock_exists.return_value = True
        
        mock_aug_instance = Mock()
        mock_aug_class.return_value = mock_aug_instance
        
        # dedup과 filtering을 건너뛰고 argumentation만 실행
        with patch('src.data_pipeline.logger'):
            pipeline.run_full_pipeline(run_dedup=False, run_filtering=False, run_argumentation=True)
        
        # final_output을 입력으로 사용했는지 확인 (하지만 run_filtering=False이므로 unique_output 사용)
        mock_aug_class.assert_called_with(str(pipeline.unique_output), batch_size=20)


class TestDataPipelineMain:
    """main 함수 테스트"""

    @patch('argparse.ArgumentParser')
    @patch('src.data_pipeline.DataPipeline')
    def test_main_status_option(self, mock_pipeline_class, mock_parser_class):
        """--status 옵션 테스트"""
        from src.data_pipeline import main
        
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.status = True
        mock_args.input = "test.csv"
        mock_args.output_dir = "./output"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock DataPipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.get_pipeline_status.return_value = {"status": "ok"}
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        with patch('src.data_pipeline.logger'):
            result = main()
        
        # 검증
        assert result == 0
        mock_pipeline_instance.get_pipeline_status.assert_called_once()
        mock_pipeline_instance.run_full_pipeline.assert_not_called()

    @patch('argparse.ArgumentParser')
    @patch('src.data_pipeline.DataPipeline')
    def test_main_full_pipeline(self, mock_pipeline_class, mock_parser_class):
        """전체 파이프라인 실행 테스트"""
        from src.data_pipeline import main
        
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.status = False
        mock_args.input = "test.csv"
        mock_args.output_dir = "./output"
        mock_args.text_col = "text"
        mock_args.skip_dedup = False
        mock_args.skip_filtering = False
        mock_args.skip_aug = False
        # 새로운 매개변수 추가
        mock_args.sample_size = 1000
        mock_args.sample_seed = 42
        mock_parser.parse_args.return_value = mock_args
        
        # Mock DataPipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        result = main()
        
        # 검증
        assert result == 0
        # 새로운 매개변수를 포함한 호출 검증
        mock_pipeline_instance.run_full_pipeline.assert_called_once_with(
            text_col="text",
            run_dedup=True,
            run_filtering=True,
            run_argumentation=True,
            sample_size=1000,
            sample_seed=42,
            filter_batch_size=mock_args.filter_batch_size,
            aug_batch_size=mock_args.aug_batch_size
        )

    @patch('argparse.ArgumentParser')
    @patch('src.data_pipeline.DataPipeline')
    def test_main_with_exception(self, mock_pipeline_class, mock_parser_class):
        """예외 발생 시 테스트"""
        from src.data_pipeline import main
        
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.status = False
        mock_args.input = "test.csv"
        mock_args.output_dir = "./output"
        mock_args.text_col = "text"
        mock_args.skip_dedup = False
        mock_args.skip_filtering = False
        mock_args.skip_aug = False
        mock_parser.parse_args.return_value = mock_args
        
        # Mock DataPipeline - 예외 발생
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run_full_pipeline.side_effect = Exception("Test error")
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        with patch('src.data_pipeline.logger') as mock_logger:
            result = main()
        
        # 검증
        assert result == 1
        mock_logger.error.assert_called()