from google import genai
from google.genai import types
import pandas as pd
from src.config.environment import get_environment_variables
from src.utils.log import logger, log_performance, decorator_log
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

env = get_environment_variables()

class DataArgumentation:
    def __init__(self, file_path: str):
        self.client = genai.Client(api_key=env.GEMINI_API_KEY)
        self.data = pd.read_parquet(file_path)
        self.prompt = f"""
당신은 스팸 문자를 분류하는 언어 모델입니다.
아래 기준에 따라 스팸여부 판정의 근거를 간단명료하게 한 문장으로 작성해 주세요.

**1. 판정 근거(한 문장):**
- **개인 정보 요구:** 신분증, 비밀번호, 카드 번호 등 개인 정보를 요구하나요?
- **기타 특이사항:** 위 항목 외에 스팸으로 의심되는 다른 패턴이 있나요?
- **발신자/수신자:** 발신 번호가 일반적이지 않거나 불분명한가요?
- **내용의 목적:** 금융 상품, 대출, 도박, 투자, 불법 복제 등의 홍보나 권유가 포함되어 있나요?
- **심리적 압박:** 긴급성, 공포, 호기심을 유발하여 즉각적인 행동을 유도하나요? (예: "기간 한정", "지금 즉시", "클릭하지 않으면 불이익")
- **링크/URL:** 일반적이지 않은 짧은 URL, 단축 URL 또는 의심스러운 링크가 포함되어 있나요?

### SPAM 문자
{{text}}

### 출력 형식
- 스팸인지 아닌지 구분을 해준다음에. 판정근거를 토대로, 스팸일 경우, 스팸으로 판정한 이유에 대해 간단 명료하게 작성해주세요.
        """

    @decorator_log(level=logging.INFO)
    @log_performance
    def data_argumentation(self):
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="SPAM AUG", unit="row"):
            text = row["CN"]
            output = []
            resp = self.client.models.generate_content(
                model=env.GEMINI_MODEL_ID,
                contents=self.prompt.format(text=text),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                )
            )
            logger.info(f"[SPAM AUG] idx={idx} success, result={resp.candidates[0].content.parts[0].text}")
            output.append(resp.candidates[0].content.parts[0].text)
        self.data.loc[idx, "output"] = output
        self.data.to_csv("./src/data/final_spam.csv", index=False)
            
if __name__ == "__main__":
    data_argumentation = DataArgumentation(file_path="./src/data/deduplicated_result.parquet")
    data_argumentation.data_argumentation()