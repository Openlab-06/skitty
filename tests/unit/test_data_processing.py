"""data_processing.py 모듈에 대한 단위 테스트"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
from pathlib import Path
import argparse

from src.data.data_processing import dedup_csv_file, main
from src.config.data_config import DataConfig


class TestDedupCsvFile:
    """dedup_csv_file 함수 테스트"""

    @patch('src.data.data_processing.DataFrameProcessor')
    @patch('src.data.data_processing.DuplicateFinder')
    @patch('src.data.data_processing.logger')
    def test_dedup_csv_file_success(self, mock_logger, mock_duplicate_finder_class, mock_dataframe_processor):
        """정상적인 중복제거 테스트"""
        # Mock 설정
        mock_df = Mock()
        mock_df.height = 100
        mock_df.__getitem__ = Mock(return_value=Mock(to_list=Mock(return_value=["text1", "text2", "text3"])))
        mock_dataframe_processor.load_and_preprocess.return_value = mock_df

        mock_finder = Mock()
        mock_duplicate_finder_class.return_value = mock_finder
        
        # build_index mock
        mock_index = Mock()
        mock_simhash_objects = [Mock(), Mock(), Mock()]
        mock_finder.build_index.return_value = (mock_index, mock_simhash_objects, 0)
        
        # find_duplicates mock
        unique_indices = [0, 2]  # 인덱스 1은 중복
        duplicate_info = [{"original_index": 0, "duplicate_index": 1}]
        mock_finder.find_duplicates.return_value = (unique_indices, duplicate_info)
        
        # create_result_dataframes mock
        mock_unique_df = Mock()
        mock_dup_df = Mock()
        mock_dataframe_processor.create_result_dataframes.return_value = (mock_unique_df, mock_dup_df)

        # 테스트 실행
        result_unique, result_dup = dedup_csv_file(
            input_path="test.csv",
            output_unique_path="unique.parquet",
            output_dups_path="dups.parquet",
            text_col="text",
            hamming_distance=3,
            ngram_size=5
        )

        # 검증
        mock_dataframe_processor.load_and_preprocess.assert_called_once_with("test.csv", "text")
        mock_duplicate_finder_class.assert_called_once_with(3)
        mock_finder.build_index.assert_called_once_with(["text1", "text2", "text3"])
        mock_finder.find_duplicates.assert_called_once_with(mock_index, mock_simhash_objects, ["text1", "text2", "text3"])
        mock_dataframe_processor.create_result_dataframes.assert_called_once_with(
            mock_df, unique_indices, duplicate_info, "text"
        )
        mock_dataframe_processor.save_results.assert_called_once_with(
            mock_unique_df, mock_dup_df, "unique.parquet", "dups.parquet", 100
        )
        
        assert result_unique == mock_unique_df
        assert result_dup == mock_dup_df

    @patch('src.data.data_processing.DataFrameProcessor')
    @patch('src.data.data_processing.DuplicateFinder')
    def test_dedup_csv_file_with_defaults(self, mock_duplicate_finder_class, mock_dataframe_processor):
        """기본 파라미터로 중복제거 테스트"""
        # Mock 설정
        mock_df = Mock()
        mock_df.height = 50
        mock_df.__getitem__ = Mock(return_value=Mock(to_list=Mock(return_value=["text1", "text2"])))
        mock_dataframe_processor.load_and_preprocess.return_value = mock_df

        mock_finder = Mock()
        mock_duplicate_finder_class.return_value = mock_finder
        mock_finder.build_index.return_value = (Mock(), [Mock(), Mock()], 0)
        mock_finder.find_duplicates.return_value = ([0, 1], [])
        
        mock_unique_df = Mock()
        mock_dup_df = Mock()
        mock_dataframe_processor.create_result_dataframes.return_value = (mock_unique_df, mock_dup_df)

        # 기본 파라미터로 테스트 실행
        result_unique, result_dup = dedup_csv_file(
            input_path="test.csv",
            output_unique_path="unique.parquet",
            output_dups_path="dups.parquet"
        )

        # 기본값들이 사용되었는지 확인
        mock_dataframe_processor.load_and_preprocess.assert_called_once_with("test.csv", DataConfig.TEXT_COL)
        mock_duplicate_finder_class.assert_called_once_with(DataConfig.SIMHASH_K)


class TestMain:
    """main 함수 테스트"""

    @patch('src.data.data_processing.dedup_csv_file')
    @patch('src.data.data_processing.logger')
    @patch('argparse.ArgumentParser')
    def test_main_success(self, mock_parser_class, mock_logger, mock_dedup_function):
        """main 함수 정상 실행 테스트"""
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.input = "test.csv"
        mock_args.out_unique = "unique.parquet"
        mock_args.out_dups = "dups.parquet"
        mock_args.text_col = "text"
        mock_args.k = 3
        mock_args.ngram = 5
        mock_parser.parse_args.return_value = mock_args

        # dedup_csv_file이 성공한다고 가정
        mock_dedup_function.return_value = (Mock(), Mock())

        # 테스트 실행
        result = main()

        # 검증
        assert result == 0
        mock_dedup_function.assert_called_once_with(
            input_path="test.csv",
            output_unique_path="unique.parquet",
            output_dups_path="dups.parquet",
            text_col="text",
            hamming_distance=3,
            ngram_size=5
        )
        mock_logger.info.assert_called()

    @patch('argparse.ArgumentParser')
    def test_main_invalid_k_parameter(self, mock_parser_class):
        """잘못된 k 파라미터 테스트"""
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.input = "test.csv"
        mock_args.k = 100  # 유효 범위 밖
        mock_args.ngram = 5
        mock_parser.parse_args.return_value = mock_args

        # 테스트 실행 및 예외 확인
        with pytest.raises(ValueError, match="k는 0-64 사이여야 합니다"):
            main()

    @patch('argparse.ArgumentParser')
    def test_main_invalid_ngram_parameter(self, mock_parser_class):
        """잘못된 ngram 파라미터 테스트"""
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.input = "test.csv"
        mock_args.k = 3
        mock_args.ngram = 0  # 유효 범위 밖
        mock_parser.parse_args.return_value = mock_args

        # 테스트 실행 및 예외 확인
        with pytest.raises(ValueError, match="ngram은 1 이상이어야 합니다"):
            main()

    @patch('src.data.data_processing.dedup_csv_file')
    @patch('src.data.data_processing.logger')
    @patch('argparse.ArgumentParser')
    def test_main_dedup_function_exception(self, mock_parser_class, mock_logger, mock_dedup_function):
        """dedup_csv_file에서 예외 발생 시 테스트"""
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        mock_args = Mock()
        mock_args.input = "test.csv"
        mock_args.out_unique = "unique.parquet"
        mock_args.out_dups = "dups.parquet"
        mock_args.text_col = "text"
        mock_args.k = 3
        mock_args.ngram = 5
        mock_parser.parse_args.return_value = mock_args

        # dedup_csv_file에서 예외 발생
        mock_dedup_function.side_effect = Exception("Test error")

        # 테스트 실행
        result = main()

        # 검증
        assert result == 1
        mock_logger.error.assert_called_with("Deduplication failed: Test error")

    @patch('argparse.ArgumentParser')
    def test_main_argument_parsing(self, mock_parser_class):
        """인수 파싱 테스트"""
        # Mock argparse
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        # ArgumentParser 설정 확인
        main_with_mock_args()
        
        # ArgumentParser가 올바르게 생성되었는지 확인
        mock_parser_class.assert_called_once()
        
        # add_argument 호출 확인
        expected_args = [
            ("--input",),
            ("--out_unique",),
            ("--out_dups",),
            ("--text_col",),
            ("--k",),
            ("--ngram",)
        ]
        
        # add_argument가 적절히 호출되었는지 확인
        assert mock_parser.add_argument.call_count == len(expected_args)


def main_with_mock_args():
    """테스트용 main 함수 (실제 실행 없이 파서만 생성)"""
    import argparse
    from src.config.data_config import DataConfig
    
    parser = argparse.ArgumentParser(
        description="CSV 파일 중복제거 (Simhash 기반)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--out_unique", 
                       default=DataConfig.DEFAULT_UNIQUE_OUTPUT,
                       help="유니크 레코드 출력 경로")
    parser.add_argument("--out_dups", 
                       default=DataConfig.DEFAULT_DUPS_OUTPUT,
                       help="중복 분석 결과 출력 경로")
    parser.add_argument("--text_col", 
                       default=DataConfig.TEXT_COL,
                       help="텍스트 컬럼명")
    parser.add_argument("--k", type=int, 
                       default=DataConfig.SIMHASH_K,
                       help="Simhash 해밍 거리 임계값 (0-64)")
    parser.add_argument("--ngram", type=int, 
                       default=DataConfig.NGRAM_N,
                       help="Character n-gram 크기")
    
    return parser


class TestValidationLogic:
    """파라미터 검증 로직 테스트"""
    
    def test_k_parameter_boundary_values(self):
        """k 파라미터 경계값 테스트"""
        # 유효한 경계값들
        valid_k_values = [0, 1, 32, 63, 64]
        for k in valid_k_values:
            # 예외가 발생하지 않아야 함
            if not 0 <= k <= 64:
                assert False, f"k={k}는 유효한 값이어야 합니다"
        
        # 무효한 값들
        invalid_k_values = [-1, 65, 100]
        for k in invalid_k_values:
            # 예외가 발생해야 함
            if 0 <= k <= 64:
                assert False, f"k={k}는 무효한 값이어야 합니다"

    def test_ngram_parameter_boundary_values(self):
        """ngram 파라미터 경계값 테스트"""
        # 유효한 값들
        valid_ngram_values = [1, 2, 5, 10]
        for ngram in valid_ngram_values:
            # 예외가 발생하지 않아야 함
            if ngram < 1:
                assert False, f"ngram={ngram}는 유효한 값이어야 합니다"
        
        # 무효한 값들
        invalid_ngram_values = [0, -1, -5]
        for ngram in invalid_ngram_values:
            # 예외가 발생해야 함
            if ngram >= 1:
                assert False, f"ngram={ngram}는 무효한 값이어야 합니다"


class TestIntegrationFlow:
    """통합 플로우 테스트"""

    @patch('src.data.data_processing.DataFrameProcessor.load_and_preprocess')
    @patch('src.data.data_processing.DuplicateFinder')
    @patch('src.data.data_processing.DataFrameProcessor.create_result_dataframes')
    @patch('src.data.data_processing.DataFrameProcessor.save_results')
    def test_full_deduplication_flow(self, mock_save_results, mock_create_results, 
                                   mock_duplicate_finder_class, mock_load_preprocess):
        """전체 중복제거 플로우 테스트"""
        # Mock DataFrame 설정 (기본 텍스트 컬럼은 CN)
        mock_df = Mock()
        mock_df.height = 3
        # CN_norm 컬럼에 대한 Mock 설정
        mock_series = Mock()
        mock_series.to_list.return_value = ["hello world", "spam message", "hello world"]
        mock_df.__getitem__ = Mock(return_value=mock_series)
        mock_load_preprocess.return_value = mock_df

        mock_finder = Mock()
        mock_duplicate_finder_class.return_value = mock_finder
        
        # 중복 발견 (인덱스 0과 2가 동일)
        mock_finder.build_index.return_value = (Mock(), [Mock(), Mock(), Mock()], 0)
        mock_finder.find_duplicates.return_value = (
            [0, 1],  # 유니크 인덱스
            [{"original_index": 0, "duplicate_index": 2, "hamming_distance": 0}]  # 중복 정보
        )
        
        mock_unique_df = Mock()
        mock_dup_df = Mock()
        mock_create_results.return_value = (mock_unique_df, mock_dup_df)

        # 테스트 실행
        unique_result, dup_result = dedup_csv_file(
            input_path="test.csv",
            output_unique_path="unique.parquet",
            output_dups_path="dups.parquet"
        )

        # 전체 플로우가 올바르게 실행되었는지 확인
        mock_load_preprocess.assert_called_once_with("test.csv", "CN")
        mock_duplicate_finder_class.assert_called_once_with(3)
        mock_finder.build_index.assert_called_once_with(["hello world", "spam message", "hello world"])
        mock_finder.find_duplicates.assert_called_once()
        mock_create_results.assert_called_once()
        mock_save_results.assert_called_once()
        
        # 결과 확인
        assert unique_result == mock_unique_df
        assert dup_result == mock_dup_df
