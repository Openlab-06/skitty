import re
from typing import List, Tuple, Dict, Optional
import polars as pl
from simhash import Simhash, SimhashIndex
from src.utils.log import logger, log_performance

# 상수 정의
SIMHASH_K = 3                  # 해밍 거리 임계값 (중복으로 묶일 최대 거리)
NGRAM_N = 5                    # char n-gram 크기
TEXT_COL = "CN"                # 입력 텍스트 컬럼명
ID_COL = "_rowid_"             # 내부 식별자

PHONE_RE = re.compile(r"\+?\d[\d\- ]{6,}\d")
URL_RE = re.compile(r"(https?://[^\s]+)")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

def normalize(text: str) -> str:
    """텍스트 정규화 함수"""
    if text is None or text == "":
        return ""
    s = str(text).strip()
    # 노이즈 정규화: 전화/URL/숫자 → 플레이스홀더
    s = PHONE_RE.sub("<PHONE>", s)
    s = URL_RE.sub("<URL>", s)
    s = NUM_RE.sub("<NUM>", s)
    # 공백/대소문자 정리
    s = re.sub(r"\s+", " ", s).lower()
    return s

def char_ngrams(s: str, n: int = NGRAM_N) -> List[str]:
    """문자 n-gram 생성"""
    L = len(s)
    if L < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(L - n + 1)]

def build_simhash(s: str, ngram_n: int = NGRAM_N) -> Simhash:
    """텍스트로부터 Simhash 객체 생성"""
    return Simhash(char_ngrams(s, ngram_n))

@log_performance
def dedup_polars_csv(
    input_path: str,
    output_unique_parquet: str,
    output_dups_parquet: str,
    text_col: str = TEXT_COL,
    k: int = SIMHASH_K,
    ngram_n: int = NGRAM_N
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    CSV 파일을 읽어 Simhash 기반 중복 제거 수행
    
    Args:
        input_path: 입력 CSV 경로
        output_unique_parquet: 유니크 레코드 저장 경로
        output_dups_parquet: 중복 분석 결과 저장 경로
        text_col: 텍스트 컬럼명
        k: Simhash 해밍 거리 임계값
        ngram_n: Character n-gram 크기
    
    Returns:
        (unique_df, dup_df) 튜플
    """
    
    # 1) CSV 읽기
    logger.info(f"Reading CSV from {input_path}")
    df = pl.read_csv(input_path)
    
    if text_col not in df.columns:
        error_msg = f"입력에 '{text_col}' 컬럼이 없습니다. 가능한 컬럼: {df.columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    original_count = df.height
    logger.info(f"Total records loaded: {original_count:,}")

    # 2) 전처리 + 내부 id 부여
    logger.debug("Starting text preprocessing")
    df = df.with_row_index(name=ID_COL).with_columns(
        pl.col(text_col)
          .cast(pl.Utf8, strict=False)
          .fill_null("")
          .map_elements(normalize, return_dtype=pl.Utf8)
          .alias(f"{text_col}_norm")
    )
    logger.debug("Text preprocessing completed")

    # 3) Simhash 객체 & 인덱스 구성
    logger.info(f"Building Simhash index with k={k}, ngram={ngram_n}")
    texts = df[f"{text_col}_norm"].to_list()
    
    # 빈 문자열 처리
    objs = []
    empty_count = 0
    for i, t in enumerate(texts):
        if t:  # 빈 문자열이 아닌 경우만 Simhash 생성
            objs.append((str(i), build_simhash(t, ngram_n)))
        else:
            # 빈 문자열은 중복으로 처리하지 않고 유지
            objs.append((str(i), Simhash([""])))  # 더미 해시
            empty_count += 1
    
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty text entries")
    
    index = SimhashIndex(objs, k=k)
    logger.debug("Simhash index built successfully")

    # 4) 인덱스로 근접 중복 묶기
    logger.info("Finding near-duplicates")
    seen = set()
    keep_ids: List[int] = []
    dup_rows: List[Dict] = []

    for i, (sid, h) in enumerate(objs):
        if sid in seen:
            continue
            
        # 빈 텍스트는 항상 유지
        if not texts[i]:
            keep_ids.append(i)
            continue
            
        keep_ids.append(i)
        near = index.get_near_dups(h)  # 문자열 id 리스트
        
        for nid in near:
            if nid == sid:
                continue
            nid_int = int(nid)
            # 빈 텍스트는 중복으로 처리하지 않음
            if not texts[nid_int]:
                continue
            if nid not in seen:
                seen.add(nid)
                dup_rows.append({
                    "original_index": i,
                    "duplicate_index": nid_int,
                    "hamming_distance": h.distance(objs[nid_int][1])  # 해밍 거리 추가
                })
    
    logger.info(f"Found {len(dup_rows)} duplicate pairs")

    # 5) 결과 DataFrame 구성
    logger.debug("Creating result DataFrames")
    
    # 유니크 레코드 (정규화 컬럼 제거)
    unique_df = df.filter(
        pl.col(ID_COL).is_in(keep_ids)
    ).drop([f"{text_col}_norm", ID_COL])
    
    # 중복 분석 DataFrame
    if dup_rows:
        dup_df = pl.DataFrame(dup_rows)
        
        # 원본 텍스트 조인 (더 효율적인 방법)
        text_mapping = df.select([
            pl.col(ID_COL),
            pl.col(text_col)
        ])
        
        # 첫 번째 조인: original_text
        dup_df = dup_df.join(
            text_mapping.select([
                pl.col(ID_COL).alias("original_index"),
                pl.col(text_col).alias("original_text")
            ]),
            on="original_index",
            how="left"
        )
        
        # 두 번째 조인: duplicate_text
        dup_df = dup_df.join(
            text_mapping.select([
                pl.col(ID_COL).alias("duplicate_index"),
                pl.col(text_col).alias("duplicate_text")
            ]),
            on="duplicate_index",
            how="left"
        )
        
        # 컬럼 순서 정리
        dup_df = dup_df.select([
            "original_index",
            "duplicate_index",
            "hamming_distance",
            "original_text",
            "duplicate_text"
        ])
    else:
        # 빈 DataFrame 생성
        dup_df = pl.DataFrame({
            "original_index": [],
            "duplicate_index": [],
            "hamming_distance": [],
            "original_text": [],
            "duplicate_text": []
        }).with_columns([
            pl.col("original_index").cast(pl.UInt32),
            pl.col("duplicate_index").cast(pl.UInt32),
            pl.col("hamming_distance").cast(pl.UInt8),
            pl.col("original_text").cast(pl.Utf8),
            pl.col("duplicate_text").cast(pl.Utf8)
        ])
        logger.info("No duplicates found")

    # 6) Parquet 저장
    logger.info("Saving results to Parquet files")
    unique_df.write_parquet(output_unique_parquet, compression="snappy")
    logger.debug(f"Unique records saved to {output_unique_parquet}")
    
    dup_df.write_parquet(output_dups_parquet, compression="snappy")
    logger.debug(f"Duplicate analysis saved to {output_dups_parquet}")

    # 7) 결과 요약
    removed_count = original_count - unique_df.height
    removal_rate = removed_count/original_count*100 if original_count > 0 else 0
    
    logger.info(f"""
Deduplication Summary:
    Original records: {original_count:,}
    Unique records: {unique_df.height:,}
    Removed duplicates: {removed_count:,} ({removal_rate:.2f}%)
    Output files:
        - Unique: {output_unique_parquet}
        - Duplicates: {output_dups_parquet}
    """.strip())
    
    return unique_df, dup_df

def main():
    """메인 실행 함수"""
    import argparse
    
    ap = argparse.ArgumentParser(
        description="CSV 파일에서 Simhash 기반 중복 제거",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    ap.add_argument("--out_unique", 
                   default="./src/data/deduplicated_result.parquet",
                   help="유니크 레코드 출력 경로")
    ap.add_argument("--out_dups", 
                   default="./src/data/duplicate_analysis.parquet",
                   help="중복 분석 결과 출력 경로")
    ap.add_argument("--text_col", default=TEXT_COL,
                   help="텍스트 컬럼명")
    ap.add_argument("--k", type=int, default=SIMHASH_K,
                   help="Simhash 해밍 거리 임계값 (0-64)")
    ap.add_argument("--ngram", type=int, default=NGRAM_N,
                   help="Character n-gram 크기")
    
    args = ap.parse_args()
    
    # 파라미터 검증
    if not 0 <= args.k <= 64:
        error_msg = f"k는 0-64 사이여야 합니다. (입력값: {args.k})"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if args.ngram < 1:
        error_msg = f"ngram은 1 이상이어야 합니다. (입력값: {args.ngram})"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"""
Starting deduplication with parameters:
    Input: {args.input}
    Text column: {args.text_col}
    Simhash k: {args.k}
    N-gram size: {args.ngram}
    """.strip())
    
    # k와 ngram 파라미터를 직접 전달
    try:
        dedup_polars_csv(
            input_path=args.input,
            output_unique_parquet=args.out_unique,
            output_dups_parquet=args.out_dups,
            text_col=args.text_col,
            k=args.k,
            ngram_n=args.ngram,
        )
        logger.info("Deduplication process completed successfully")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        import traceback
        logger.debug(f"Stack trace:\n{traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())