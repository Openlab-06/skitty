"""
데이터 처리 파이프라인 전체 오케스트레이션
"""
from pathlib import Path
from typing import Optional
import polars as pl

from src.config.data_config import DataConfig
from src.utils.log import logger, log_performance
from src.data.data_processing import dedup_csv_file
from src.data.data_argumentation import DataArgumentation
from src.data.data_filtering import DataFiltering

class DataPipeline:
    """전체 데이터 처리 파이프라인을 관리하는 클래스"""
    
    def __init__(self, input_csv_path: str, output_dir: str = "./src/data"):
        self.input_csv_path = Path(input_csv_path)
        self.output_dir = Path(output_dir)
        
        # 출력 파일 경로 설정
        self.unique_output = self.output_dir / "deduplicated_result.parquet"
        self.dups_output = self.output_dir / "duplicate_analysis.parquet"
        self.final_output = self.output_dir / "final_spam.csv"
        
    @log_performance
    def run_full_pipeline(self, 
                         text_col: str = DataConfig.TEXT_COL,
                         run_dedup: bool = True,
                         run_filtering: bool = True,
                         run_argumentation: bool = True,
                         sample_size: Optional[float] = None,
                         sample_seed: Optional[int] = None) -> None:
        """
        전체 데이터 처리 파이프라인 실행
        
        Args:
            text_col: 텍스트 컬럼명
            run_dedup: 중복제거 실행 여부
            run_filtering: 데이터 필터링 실행 여부
            run_argumentation: 데이터 증강 실행 여부
            sample_size: 필터링용 샘플링 비율 (기본값: DataFilteringConfig.DEFAULT_SAMPLE_SIZE)
            sample_seed: 샘플링 시드 (기본값: DataFilteringConfig.DEFAULT_SAMPLE_SEED)
        """
        logger.info("Starting full data pipeline")
        
        try:
            # 1. 중복제거 단계
            if run_dedup:
                logger.info("=== Step 1: Deduplication ===")
                unique_df, dup_df = dedup_csv_file(
                    input_path=str(self.input_csv_path),
                    output_unique_path=str(self.unique_output),
                    output_dups_path=str(self.dups_output),
                    text_col=text_col
                )
                logger.info(f"Deduplication completed. Unique records: {unique_df.height:,}")
            else:
                logger.info("Skipping deduplication step")
                if not self.unique_output.exists():
                    raise FileNotFoundError(f"Deduplicated file not found: {self.unique_output}")
            
            # 2. 데이터 필터링 단계
            if run_filtering:
                logger.info("=== Step 2: Data Filtering ===")
                data_filter = DataFiltering(
                    str(self.unique_output), 
                    sample_size=sample_size, 
                    sample_seed=sample_seed
                )
                data_filter.data_filter()
                logger.info(f"Data filtering completed. Output: {self.final_output}")
            else:
                logger.info("Skipping data filtering step")
            
            # 3. 데이터 증강 단계
            if run_argumentation:
                logger.info("=== Step 3: Data Argumentation ===")
                # 필터링이 실행된 경우 필터링 결과를 사용, 아니면 중복제거 결과 사용
                input_file = str(self.final_output) if run_filtering and self.final_output.exists() else str(self.unique_output)
                data_aug = DataArgumentation(input_file)
                data_aug.data_argumentation()
                logger.info(f"Data argumentation completed. Output: {self.final_output}")
            else:
                logger.info("Skipping data argumentation step")
            
            logger.info("Full pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_pipeline_status(self) -> dict:
        """파이프라인 상태 확인"""
        status = {
            "input_exists": self.input_csv_path.exists(),
            "deduplicated_exists": self.unique_output.exists(),
            "duplicates_analysis_exists": self.dups_output.exists(),
            "final_output_exists": self.final_output.exists(),
        }
        
        if status["input_exists"]:
            # 입력 파일 크기 정보
            input_size = self.input_csv_path.stat().st_size / (1024**2)  # MB
            status["input_size_mb"] = round(input_size, 2)
        
        if status["final_output_exists"]:
            # 최종 출력 파일 크기 정보
            final_size = self.final_output.stat().st_size / (1024**2)  # MB
            status["final_size_mb"] = round(final_size, 2)
        
        return status


def main():
    """CLI 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="전체 데이터 처리 파이프라인 실행",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--output_dir", default="./src/data", help="출력 디렉토리")
    parser.add_argument("--text_col", default=DataConfig.TEXT_COL, help="텍스트 컬럼명")
    parser.add_argument("--skip_dedup", action="store_true", help="중복제거 건너뛰기")
    parser.add_argument("--skip_filtering", action="store_true", help="데이터 필터링 건너뛰기")
    parser.add_argument("--skip_aug", action="store_true", help="데이터 증강 건너뛰기")
    parser.add_argument("--status", action="store_true", help="파이프라인 상태 확인만")
    parser.add_argument("--sample_size", type=float, help="필터링용 샘플링 비율 (예: 0.02)")
    parser.add_argument("--sample_seed", type=int, help="샘플링 시드 (예: 42)")
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.input, args.output_dir)
    
    if args.status:
        status = pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        return 0
    
    try:
        pipeline.run_full_pipeline(
            text_col=args.text_col,
            run_dedup=not args.skip_dedup,
            run_filtering=not args.skip_filtering,
            run_argumentation=not args.skip_aug,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
