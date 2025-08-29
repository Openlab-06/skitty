import re
from typing import List, Tuple, Dict
import polars as pl
from simhash import Simhash, SimhashIndex
from src.config.data_config import DataConfig
from src.utils.log import logger, log_performance, decorator_log
from src.data.data_normalize import TextProcessor
from src.utils.exception import DataDeduplicationError
import logging

class SimhashGenerator:
    """Simhash 생성을 담당하는 클래스"""
    
    def __init__(self, ngram_size: int = DataConfig.NGRAM_N):
        self.ngram_size = ngram_size
        self.text_processor = TextProcessor()
    
    def create_simhash(self, text: str) -> Simhash:
        """텍스트로부터 Simhash 생성"""
        ngrams = self.text_processor.create_char_ngrams(text, self.ngram_size)
        return Simhash(ngrams)

class DuplicateFinder:
    """중복 찾기를 담당하는 클래스"""
    
    def __init__(self, hamming_distance: int = DataConfig.SIMHASH_K):
        self.hamming_distance = hamming_distance
        self.simhash_gen = SimhashGenerator()
    
    @decorator_log(level=logging.INFO)
    @log_performance
    def build_index(self, texts: List[str]) -> Tuple[SimhashIndex, List[Tuple[str, Simhash]], int]:
        """Simhash 인덱스 구축"""
        logger.info(f"Building Simhash index (k={self.hamming_distance})")
        
        simhash_objects = []
        empty_count = 0
        
        for i, text in enumerate(texts):
            if text:
                simhash_obj = self.simhash_gen.create_simhash(text)
            else:
                simhash_obj = Simhash([""])  # 빈 텍스트용 더미
                empty_count += 1
            
            simhash_objects.append((str(i), simhash_obj))
        
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty text entries")
        
        try:
            index = SimhashIndex(simhash_objects, k=self.hamming_distance)
            return index, simhash_objects, empty_count
        except Exception as e:
            raise DataDeduplicationError(error=str(e), status_code=500) from e

    
    @decorator_log(level=logging.INFO)
    @log_performance
    def find_duplicates(self, index: SimhashIndex, simhash_objects: List[Tuple[str, Simhash]], 
                       texts: List[str]) -> Tuple[List[int], List[Dict]]:
        """중복 찾기"""
        logger.info("Finding near-duplicates")
        
        seen_duplicates = set()
        unique_indices = []
        duplicate_info = []
        
        try:
            for i, (string_id, simhash_obj) in enumerate(simhash_objects):
                if string_id in seen_duplicates:
                    continue
                
                # 빈 텍스트는 항상 유지
                if not texts[i]:
                    unique_indices.append(i)
                    continue
                
                unique_indices.append(i)  # 이 레코드는 유지
                
                # 유사한 레코드들 찾기
                similar_ids = index.get_near_dups(simhash_obj)
                
                for similar_id in similar_ids:
                    if similar_id == string_id:  # 자기 자신은 제외
                        continue
                    
                    similar_idx = int(similar_id)
                    
                    # 빈 텍스트는 중복으로 처리하지 않음
                    if not texts[similar_idx]:
                        continue
                    
                    if similar_id not in seen_duplicates:
                        seen_duplicates.add(similar_id)
                        
                        # 중복 정보 저장
                        hamming_dist = simhash_obj.distance(simhash_objects[similar_idx][1])
                        duplicate_info.append({
                            "original_index": i,
                            "duplicate_index": similar_idx,
                            "hamming_distance": hamming_dist
                        })
            logger.info(f"Found {len(duplicate_info)} duplicate pairs")
            return unique_indices, duplicate_info
        except Exception as e:  
            raise DataDeduplicationError(error=str(e), status_code=500) from e
