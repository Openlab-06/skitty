class DeduplicationConfig:
    # Simhash 설정
    SIMHASH_K = 3                  # 해밍 거리 임계값
    NGRAM_N = 5                    # char n-gram 크기
    
    # 컬럼 설정
    TEXT_COL = "CN"               # 입력 텍스트 컬럼명
    ID_COL = "_rowid_"            # 내부 식별자
    
    # 정규화 패턴
    PHONE_PATTERN = r"\+?\d[\d\- ]{6,}\d"
    URL_PATTERN = r"(https?://[^\s]+)"
    NUM_PATTERN = r"\b\d+(?:[.,]\d+)?\b"
    
    # 출력 설정
    DEFAULT_UNIQUE_OUTPUT = "./src/data/deduplicated_result.parquet"
    DEFAULT_DUPS_OUTPUT = "./src/data/duplicate_analysis.parquet"
    COMPRESSION = "snappy"