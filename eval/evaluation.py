import asyncio
import logging
from datasets import load_dataset
from openai import AsyncOpenAI
from opik.evaluation.metrics import SentenceBLEU, ROUGE
from src.config.env_config import get_config
from src.utils.log import logger, async_log_performance, async_decorator_log

env = get_config()


def safe_mean(values: list[float]) -> float:
    """빈 리스트일 경우 0.0 반환"""
    return sum(values) / len(values) if values else 0.0


class LLMEvaluation:
    def __init__(self, limit: int | None = None):
        self.dataset = load_dataset("Devocean-06/Spam_QA-Corpus", split="test")

        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        self.bleu = SentenceBLEU()
        self.rouge = ROUGE()

        self.model = AsyncOpenAI(base_url=env.SPAM_MODEL_URL, api_key=env.SPAM_MODEL_API_KEY)
        self.model_name = env.SPAM_MODEL

    def _build_prompt(self, sample: dict) -> tuple[str, str]:
        """Alpaca 포맷(instruction/input/output) → 프롬프트/정답 생성"""
        instruction = sample["instruction"].strip()
        input_text = sample["input"].strip()
        reference = sample["output"].strip()

        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction

        return prompt, reference

    @async_decorator_log(level=logging.DEBUG)
    @async_log_performance
    async def evaluate(self):
        bleu_scores: list[float] = []
        rouge_scores: list[dict] = []

        for i, sample in enumerate(self.dataset):
            try:
                prompt, reference = self._build_prompt(sample)

                resp = await self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                prediction = resp.choices[0].message.content.strip()

                bleu = self.bleu.compute(prediction=prediction, references=[reference])
                bleu_scores.append(float(bleu))

                rouge = self.rouge.compute(prediction=prediction, references=[reference])
                rouge_scores.append({k: float(v) for k, v in rouge.items()})

            except Exception as e:
                logger.warning("샘플 %d 처리 스킵: %s", i, e)

        if not bleu_scores:
            raise RuntimeError("유효한 샘플이 없어 점수를 계산할 수 없습니다.")

        avg_bleu = safe_mean(bleu_scores)
        rouge_keys = rouge_scores[0].keys()
        avg_rouge = {k: safe_mean([r[k] for r in rouge_scores]) for k in rouge_keys}

        return {"BLEU": avg_bleu, "ROUGE": avg_rouge}