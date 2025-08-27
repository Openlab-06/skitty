import re
from typing import List
from src.config.data_config import DeduplicationConfig

class TextProcessor:
    """텍스트 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.phone_re = re.compile(DeduplicationConfig.PHONE_PATTERN)
        self.url_re = re.compile(DeduplicationConfig.URL_PATTERN)
        self.num_re = re.compile(DeduplicationConfig.NUM_PATTERN)
    
    def normalize(self, text: str) -> str:
        """텍스트 정규화"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 노이즈를 플레이스홀더로 변경
        text = self.phone_re.sub("<PHONE>", text)
        text = self.url_re.sub("<URL>", text)
        text = self.num_re.sub("<NUM>", text)

        # 공백 정리 및 소문자 변환
        text = re.sub(r"\s+", " ", text).lower()
        return text
    
    def create_char_ngrams(self, text: str, n: int) -> List[str]:
        """문자 n-gram 생성"""
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text) - n + 1)]