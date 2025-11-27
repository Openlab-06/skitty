import asyncio
import logging
from typing import Any, List, Tuple

import pandas as pd
from google import genai
from openai import AsyncOpenAI
from tqdm import tqdm

from src.config.env_config import get_config
from src.data import constants
from src.utils.enum import DataComplexity
from src.utils.exception import DataFilteringError
from src.utils.log import logger, log_performance, decorator_log

env = get_config()
logging.basicConfig(level=logging.INFO)

class DataFiltering:
    def __init__(self, file_path: str, sample_size: float = None, sample_seed: int = None, batch_size: int = constants.DEFAULT_FILTER_BATCH_SIZE):
        self.gemini_client = genai.Client(api_key=env.GEMINI_API_KEY)
        self.openai_client = AsyncOpenAI(api_key=env.OPENAI_API_KEY)
        self.batch_size = batch_size  # 배치 크기 설정

        frac = sample_size if sample_size is not None else constants.DEFAULT_SAMPLE_SIZE
        seed = sample_seed if sample_seed is not None else constants.DEFAULT_SAMPLE_SEED

        df = pd.read_parquet(file_path)
        logger.info(f"Loaded rows (before sample): {len(df)}")
        self.data = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
        logger.info(f"Sample frac={frac}, seed={seed} -> rows={len(self.data)}")
        logger.info(f"Data columns: {self.data.columns.tolist()}")

        if "CN" not in self.data.columns:
            raise DataFilteringError(error="Input data must contain 'CN' column.", status_code=400)

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
""".strip()

    async def _process_single_text(self, text: str, idx: int) -> str:
        """단일 텍스트 처리 (Gemini → OpenAI failover)"""
        prompt = self.prompt_template.format(text=text)

        try:
            # 1) Gemini 시도
            gm_resp = await self.gemini_client.aio.models.generate_content(
                model=env.GEMINI_MODEL_FILTER,
                contents=prompt,
                config={
                    "response_mime_type": "text/x.enum",
                    "response_schema": DataComplexity,
                },
            )

            result_text = getattr(gm_resp, "text", None)
            if not result_text:
                try:
                    result_text = gm_resp.candidates[0].content.parts[0].text
                except Exception:
                    raise RuntimeError("Empty Gemini enum response")

            logger.debug(f"[SPAM FILTER] idx={idx} Gemini success: {result_text}")
            return result_text

        except Exception as gem_e:
            logger.warning(f"[SPAM FILTER] Gemini failed at idx={idx}: {gem_e} -> fallback to OpenAI")

            # 2) OpenAI 시도
            try:
                response = await self.openai_client.chat.completions.create(
                    model=env.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50  # 짧은 응답만 필요
                )
                result = response.choices[0].message.content.strip()
                logger.debug(f"[SPAM FILTER] idx={idx} OpenAI success: {result}")
                return result

            except Exception as e:
                logger.error(f"[SPAM FILTER] idx={idx} failed after Gemini+OpenAI: {e}")
                raise DataFilteringError(error=str(e), status_code=500) from e

    async def _process_batch(self, batch_data: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """배치 처리"""
        tasks = [
            self._process_single_text(text, idx) 
            for idx, text in batch_data
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for (idx, text), result in zip(batch_data, results):
            if isinstance(result, Exception):
                logger.error(f"[BATCH] idx={idx} failed: {result}")
                # 실패한 경우 기본값 사용
                processed_results.append((idx, "MEDIUM"))
            else:
                processed_results.append((idx, result))
        
        return processed_results

    @decorator_log(level=logging.INFO)
    @log_performance
    async def data_filter(self):
        """배치 처리를 통한 데이터 필터링"""
        logger.info(f"Starting batch processing with batch_size={self.batch_size}")
        
        # 결과를 저장할 딕셔너리 (인덱스별로 결과 저장)
        results_dict = {}
        
        # 데이터를 배치로 나누기
        total_rows = len(self.data)
        batch_data = []
        
        for idx, row in self.data.iterrows():
            text = row["CN"]
            batch_data.append((idx, text))
            
            # 배치가 찼거나 마지막 배치인 경우 처리
            if len(batch_data) == self.batch_size or idx == total_rows - 1:
                logger.info(f"Processing batch of {len(batch_data)} items (total processed: {len(results_dict)}/{total_rows})")
                
                try:
                    batch_results = await self._process_batch(batch_data)
                    
                    # 결과 저장
                    for batch_idx, result in batch_results:
                        results_dict[batch_idx] = result
                        logger.debug(f"[BATCH] idx={batch_idx} result={result}")
                    
                    logger.info(f"Batch completed successfully. Results: {len(batch_results)}")
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # 배치 실패 시 개별 처리로 fallback
                    logger.info("Falling back to individual processing for this batch")
                    for batch_idx, text in batch_data:
                        try:
                            result = await self._process_single_text(text, batch_idx)
                            results_dict[batch_idx] = result
                        except Exception as individual_e:
                            logger.error(f"Individual processing failed for idx={batch_idx}: {individual_e}")
                            results_dict[batch_idx] = "MEDIUM"  # 기본값
                
                # 배치 초기화
                batch_data = []
        
        # 결과를 순서대로 정렬하여 리스트로 변환
        output_list = [results_dict[idx] for idx in sorted(results_dict.keys())]
        
        logger.info(f"Processing completed. Total results: {len(output_list)}")
        
        # 데이터프레임에 결과 추가
        self.data["complexity"] = output_list

        self.data.to_csv("./src/data/filter_spam.csv", index=False)
        logger.info("Saved filtered data to ./src/data/filter_spam.csv")

if __name__ == "__main__":
    dfilt = DataFiltering(file_path="./src/data/deduplicated_result.parquet")
    logger.info(f"Sampled data size: {len(dfilt.data)}")
    asyncio.run(dfilt.data_filter())
