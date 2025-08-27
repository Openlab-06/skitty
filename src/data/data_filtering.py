from google import genai
from google.genai import types
import pandas as pd
from src.config.env_config import get_config
from src.utils.log import logger, log_performance, decorator_log
import logging
from tqdm import tqdm
from enum import Enum
import time

env = get_config()

class DataComplexity(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    VERY_HIGH = 3
    EXTREMELY_HIGH = 4

class DataFiltering:
    def __init__(self, file_path: str):
        self.client = genai.Client(api_key=env.GEMINI_API_KEY)
        self.data = pd.read_parquet(file_path).sample(frac=0.02, random_state=42).reset_index(drop=True)
        print(f"Data columns: {self.data.columns.tolist()}")
        print(f"Data shape: {self.data.shape}")
        
        self.prompt_template = """
        현재 주어진 텍스트는 스팸 문자 데이터입니다.
        주어진 텍스트를 분석해서, 복잡도를 기준으로 스팸 문자로 판별하기 위한 점수를 출력해주세요.
        
        복잡도 기준:
        - LOW: 단순한 텍스트
        - MEDIUM: 보통 복잡도
        - HIGH: 복잡한 텍스트
        - VERY_HIGH: 매우 복잡한 텍스트
        - EXTREMELY_HIGH: 극도로 복잡한 텍스트
        
        ### SPAM 문자
        {text}
        
        복잡도만 응답하세요: LOW, MEDIUM, HIGH, VERY_HIGH, EXTREMELY_HIGH 중 하나
        """

    @decorator_log(level=logging.INFO)
    @log_performance
    def data_filter(self):
        output_list = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="SPAM FILTER", unit="row"):
            text = row["CN"]
            try:
                prompt = self.prompt_template.format(text=text)  # 템플릿에 텍스트 삽입
                
                resp = self.client.models.generate_content(
                    model=env.GEMINI_MODEL_FILTER,
                    contents=prompt,  # 포맷된 프롬프트 사용
                    config={
                        'response_mime_type': 'text/x.enum',  # enum 대신 json 사용
                        'response_schema': DataComplexity
                    },
                )
                result = resp.text
                logger.info(f"[SPAM FILTER] idx={idx} text_preview={text[:50]}... result={result}")
                output_list.append(result)
                
            except Exception as e:
                logger.error(f"[SPAM FILTER] idx={idx} failed: {e}")
                output_list.append("LOW")  # 기본값 설정
        
        self.data["complexity"] = output_list
        time.sleep(1)
        self.data.to_csv("./src/data/filter_spam.csv", index=False)

if __name__ == "__main__":
    dfilt = DataFiltering(file_path="./src/data/deduplicated_result.parquet")
    print(f"Sampled data size: {len(dfilt.data)}")
    
    dfilt.data_filter()
