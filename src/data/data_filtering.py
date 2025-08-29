from google import genai
from google.genai import types
import pandas as pd
from src.config.env_config import get_config
from src.config.data_config import DataConfig   
from src.utils.log import logger, log_performance, decorator_log
import logging
from tqdm import tqdm
from src.domain.data_enum import DataComplexity
import asyncio
from src.utils.exception import DataFilteringError

env = get_config()

class DataFiltering:
    def __init__(self, file_path: str, sample_size: float = None, sample_seed: int = None):
        self.client = genai.Client(api_key=env.GEMINI_API_KEY)
        
        # 파라미터가 제공되지 않으면 기본값 사용
        frac = sample_size if sample_size is not None else DataConfig.DEFAULT_SAMPLE_SIZE
        seed = sample_seed if sample_seed is not None else DataConfig.DEFAULT_SAMPLE_SEED
        
        self.data = pd.read_parquet(file_path).sample(frac=frac, random_state=seed).reset_index(drop=True)
        logger.info(f"Data columns: {self.data.columns.tolist()}")
        logger.info(f"Data shape: {self.data.shape}")
        
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
    async def data_filter(self):
        output_list = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="SPAM FILTER", unit="row"):
            text = row["CN"]
            try:
                prompt = self.prompt_template.format(text=text)  # 템플릿에 텍스트 삽입
                
                resp = await self.client.aio.models.generate_content(
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
                raise DataFilteringError(error=str(e), status_code=500) from e
        
        self.data["complexity"] = output_list
        #self.data["complexity"] = self.data["complexity"].astype(int)

        #self.data = self.data[self.data["complexity"] >= 3]
        self.data.to_csv("./src/data/filter_spam.csv", index=False)

if __name__ == "__main__":
    dfilt = DataFiltering(file_path="./src/data/deduplicated_result.parquet")
    logger.info(f"Sampled data size: {len(dfilt.data)}")
    asyncio.run(dfilt.data_filter())