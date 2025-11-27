import pandas as pd
from google import genai
from google.genai import types
from src.utils.log import logger, log_performance, decorator_log
from src.utils.exception import DataAugmentationError
from src.config.env_config import get_config
from src.data import constants
import logging
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
from typing import List, Tuple

env = get_config()

class DataAugmentation:
    def __init__(self, file_path: str, batch_size: int = constants.DEFAULT_AUG_BATCH_SIZE):
        self.gemini_client = genai.Client(api_key=env.GEMINI_API_KEY)
        self.openai_client = AsyncOpenAI(api_key=env.OPENAI_API_KEY)
        self.data = pd.read_csv(file_path)
        self.batch_size = batch_size  # 증강은 더 긴 응답이므로 작은 배치 사용
        self.prompt ="""
당신은 스팸 문자로 판정한 근거를 생성하는 대형 언어 모델입니다.
아래 기준에 따라 스팸여부 판정의 근거를 간단명료하게 한 문장으로 작성해 주세요. 출력 포맷은 XAI 설명에 적합하도록 일관성 있게 템플릿 형식으로 고정되어야 하며, 스팸 여부 및 그 근거를 명쾌하게 제시해야 합니다.

**1. 판정 근거(한 문장, 템플릿):**
- **개인 정보 요구:** 신분증, 비밀번호, 카드 번호 등 개인 정보를 요구했기 때문입니다.
- **기타 특이사항:** 위 항목 외에 스팸으로 의심되는 다른 패턴이 있습니다.
- **발신자/수신자:** 발신 번호가 일반적이지 않거나 불분명하기 때문입니다.
- **내용의 목적:** 금융 상품, 대출, 도박, 투자, 불법 복제 등의 홍보나 권유가 포함되어 있기 때문입니다.
- **심리적 압박:** 긴급성, 공포, 호기심을 유발하여 즉각적인 행동을 유도했기 때문입니다. (예: "기간 한정", "지금 즉시", "클릭하지 않으면 불이익")
- **링크/URL:** 일반적이지 않은 짧은 URL, 단축 URL 또는 의심스러운 링크가 포함되어 있기 때문입니다.

**2. 필수 조건**
- 반드시 출력 형식에 따라서 [스팸 판정 이유] 템플릿을 사용해야 합니다.
- 스팸으로 판정한 이유에 대해서 구체적인 이유로 100자 이상으로 설명해야 합니다.
- 반드시 위 판정 근거를 먼저 언급한 뒤에 출력 형식에 맞게 스팸 판정 이유를 생성해야 합니다.
- 스팸 판정 이유 생성 시, 위 스팸 문자는 ~~ 으로 시작해야합니다.
- 그리고 전제조건은 모두 스팸 문자로 분류된 형식이니 스팸이 아니라고 언급하면 안됩니다.

### SPAM 문자
{text}

### 출력 형식 예시
- 판정 근거 : 개인정보 요구
- 스팸 판정 이유: 위 스팸 문자는 개인정보를 요구하는 스팸으로 아파트 분양 및 부동산 투자 권유가 포함되어 있으며, 긴급성을 강조하여 즉각적인 행동을 유도하고 있습니다.
"""

    async def _process_single_text(self, text: str, idx: int) -> str:
        """단일 텍스트 처리 (Gemini → OpenAI failover)"""
        try:
            # Gemini 시도
            result = await self.gemini_client.aio.models.generate_content(
                model=env.GEMINI_MODEL_ARGU,
                contents=self.prompt.format(text=text),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            
            logger.debug(f"[SPAM AUG] idx={idx} Gemini success: {result.text[:100]}...")
            return result.text

        except Exception as e:
            logger.warning(f"[SPAM AUG] idx={idx} Gemini failed: {e} -> fallback to OpenAI")

            # OpenAI 시도
            try:
                result = await self.openai_client.chat.completions.create(
                    model=env.OPENAI_MODEL,
                    messages=[{"role": "user", "content": self.prompt.format(text=text)}],
                    temperature=0.0,
                    max_tokens=512
                )
                output = result.choices[0].message.content
                logger.debug(f"[SPAM AUG] idx={idx} OpenAI success: {output[:100]}...")
                return output

            except Exception as e:
                logger.error(f"[SPAM AUG] idx={idx} failed after all attempts: {e}")
                raise DataAugmentationError(error=str(e), status_code=500) from e

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
                logger.error(f"[BATCH AUG] idx={idx} failed: {result}")
                # 실패한 경우 기본 설명 사용
                processed_results.append((idx, f"이 텍스트는 스팸으로 분류되었습니다: {text[:50]}..."))
            else:
                processed_results.append((idx, result))
        
        return processed_results

    @decorator_log(level=logging.INFO)
    @log_performance
    async def data_argumentation(self):
        """배치 처리를 통한 데이터 증강"""
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
                        logger.debug(f"[BATCH AUG] idx={batch_idx} result={result[:100]}...")
                    
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
                            results_dict[batch_idx] = f"이 텍스트는 스팸으로 분류되었습니다: {text[:50]}..."  # 기본값
                
                # 배치 초기화
                batch_data = []
        
        # 결과를 순서대로 정렬하여 리스트로 변환
        output_list = [results_dict[idx] for idx in sorted(results_dict.keys())]
        
        logger.info(f"Processing completed. Total results: {len(output_list)}")
        
        # 데이터프레임에 결과 추가
        self.data["output"] = output_list
        self.data.to_csv("./src/data/final_spam.csv", index=False)
        logger.info("Saved augmented data to ./src/data/final_spam.csv")
            
if __name__ == "__main__":
    data_augmentation = DataAugmentation(file_path="./src/data/filter_spam_high.csv")
    asyncio.run(data_augmentation.data_argumentation())