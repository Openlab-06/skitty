from src.config.data_config import DataConfig
from src.utils.log import logger, log_performance
from src.data.data_io import DataFrameProcessor
from src.data.data_dedup import DuplicateFinder
from typing import Tuple
import polars as pl

@log_performance
def dedup_csv_file(input_path: str, output_unique_path: str, output_dups_path: str,
                  text_col: str = DataConfig.TEXT_COL,
                  hamming_distance: int = DataConfig.SIMHASH_K,
                  ngram_size: int = DataConfig.NGRAM_N) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    CSV 파일 중복제거 메인 함수
    
    Args:
        input_path: 입력 CSV 파일 경로
        output_unique_path: 유니크 레코드 저장 경로
        output_dups_path: 중복 분석 결과 저장 경로
        text_col: 텍스트 컬럼명
        hamming_distance: Simhash 해밍 거리 임계값
        ngram_size: Character n-gram 크기
    
    Returns:
        (unique_df, duplicate_df) 튜플
    """
    
    # 1. 데이터 로드 및 전처리
    df = DataFrameProcessor.load_and_preprocess(input_path, text_col)
    original_count = df.height
    
    # 2. 중복 찾기
    duplicate_finder = DuplicateFinder(hamming_distance)
    texts = df[f"{text_col}_norm"].to_list()
    
    index, simhash_objects, empty_count = duplicate_finder.build_index(texts)
    unique_indices, duplicate_info = duplicate_finder.find_duplicates(index, simhash_objects, texts)
    
    # 3. 결과 DataFrame 생성
    unique_df, dup_df = DataFrameProcessor.create_result_dataframes(
        df, unique_indices, duplicate_info, text_col
    )
    
    # 4. 결과 저장
    DataFrameProcessor.save_results(
        unique_df, dup_df, output_unique_path, output_dups_path, original_count
    )
    
    return unique_df, dup_df

def main():
    """CLI 실행 함수"""
    import argparse
    
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
    
    args = parser.parse_args()
    
    # 파라미터 검증
    if not 0 <= args.k <= 64:
        raise ValueError(f"k는 0-64 사이여야 합니다. (입력값: {args.k})")
    if args.ngram < 1:
        raise ValueError(f"ngram은 1 이상이어야 합니다. (입력값: {args.ngram})")
    
    logger.info(f"""
Starting deduplication:
    Input: {args.input}
    Text column: {args.text_col}
    Hamming distance: {args.k}
    N-gram size: {args.ngram}
    """.strip())
    
    try:
        dedup_csv_file(
            input_path=args.input,
            output_unique_path=args.out_unique,
            output_dups_path=args.out_dups,
            text_col=args.text_col,
            hamming_distance=args.k,
            ngram_size=args.ngram
        )
        logger.info("Deduplication completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())