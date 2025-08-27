from typing import List, Dict, Tuple
import polars as pl
from src.config.data_config import DataConfig
from src.utils.log import logger
from src.data.data_normalize import TextProcessor

class DataFrameProcessor:
    """DataFrame 처리를 담당하는 클래스"""
    
    @staticmethod
    def load_and_preprocess(input_path: str, text_col: str) -> pl.DataFrame:
        """CSV 로드 및 전처리"""
        logger.info(f"Reading CSV from {input_path}")
        df = pl.read_csv(input_path)
        
        if text_col not in df.columns:
            raise ValueError(f"'{text_col}' 컬럼이 없습니다. 가능한 컬럼: {df.columns}")
        
        logger.info(f"Total records loaded: {df.height:,}")
        
        # 전처리
        processor = TextProcessor()
        df = df.with_row_index(name=DataConfig.ID_COL).with_columns(
            pl.col(text_col)
              .cast(pl.Utf8, strict=False)
              .fill_null("")
              .map_elements(processor.normalize, return_dtype=pl.Utf8)
              .alias(f"{text_col}_norm")
        )
        
        return df
    
    @staticmethod
    def create_result_dataframes(df: pl.DataFrame, unique_indices: List[int], 
                               duplicate_info: List[Dict], text_col: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """결과 DataFrame 생성"""
        logger.debug("Creating result DataFrames")
        
        # 유니크 레코드
        unique_df = df.filter(
            pl.col(DataConfig.ID_COL).is_in(unique_indices)
        ).drop([f"{text_col}_norm", DataConfig.ID_COL])
        
        # 중복 분석 결과
        if duplicate_info:
            dup_df = pl.DataFrame(duplicate_info)
            
            # 원본 텍스트 정보 추가
            text_mapping = df.select([
                pl.col(DataConfig.ID_COL),
                pl.col(text_col)
            ])
            
            # 원본 텍스트 조인
            dup_df = dup_df.join(
                text_mapping.select([
                    pl.col(DataConfig.ID_COL).alias("original_index"),
                    pl.col(text_col).alias("original_text")
                ]),
                on="original_index",
                how="left"
            )
            
            # 중복 텍스트 조인
            dup_df = dup_df.join(
                text_mapping.select([
                    pl.col(DataConfig.ID_COL).alias("duplicate_index"),
                    pl.col(text_col).alias("duplicate_text")
                ]),
                on="duplicate_index",
                how="left"
            )
            
            # 컬럼 순서 정리
            dup_df = dup_df.select([
                "original_index", "duplicate_index", "hamming_distance",
                "original_text", "duplicate_text"
            ])
        else:
            # 빈 DataFrame
            dup_df = pl.DataFrame({
                "original_index": [],
                "duplicate_index": [],
                "hamming_distance": [],
                "original_text": [],
                "duplicate_text": []
            }, schema={
                "original_index": pl.UInt32,
                "duplicate_index": pl.UInt32,
                "hamming_distance": pl.UInt8,
                "original_text": pl.Utf8,
                "duplicate_text": pl.Utf8
            })
            logger.info("No duplicates found")
        
        return unique_df, dup_df
    
    @staticmethod
    def save_results(unique_df: pl.DataFrame, dup_df: pl.DataFrame,
                    unique_path: str, dup_path: str, original_count: int):
        """결과 저장 및 요약"""
        logger.info("Saving results to Parquet files")
        
        unique_df.write_parquet(unique_path, compression=DataConfig.COMPRESSION)
        dup_df.write_parquet(dup_path, compression=DataConfig.COMPRESSION)
        
        # 결과 요약
        removed_count = original_count - unique_df.height
        removal_rate = removed_count / original_count * 100 if original_count > 0 else 0
        
        logger.info(f"""
Deduplication Summary:
    Original records: {original_count:,}
    Unique records: {unique_df.height:,}
    Removed duplicates: {removed_count:,} ({removal_rate:.2f}%)
    Output files:
        - Unique: {unique_path}
        - Duplicates: {dup_path}
        """.strip())