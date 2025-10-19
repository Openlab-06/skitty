import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from openai import AsyncOpenAI
from opik.evaluation.metrics import SentenceBLEU
from sentence_transformers import CrossEncoder
import numpy as np
from src.config.env_config import get_config
from src.utils.log import logger, async_log_performance, async_decorator_log

env = get_config()


def safe_mean(values: list[float]) -> float:
    """빈 리스트일 경우 0.0 반환"""
    return sum(values) / len(values) if values else 0.0


@dataclass
class EvaluationMetrics:
    """평가 메트릭 데이터 클래스"""
    sample_id: int
    prompt: str
    reference: str
    prediction: str
    bleu_score: float
    semantic_similarity: float
    llm_judge_score: float
    llm_judge_explanation: str


class LLMEvaluation:
    def __init__(self, limit: int | None = None, model_name: str = "google/gemma-3-4b-it"):
        self.dataset = load_dataset("Devocean-06/Spam_QA-Corpus", split="test")

        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        # 메트릭 초기화
        self.bleu = SentenceBLEU()
        # Semantic Similarity용 CrossEncoder (스팸 설명 품질 평가)
        self.semantic_model = CrossEncoder("cross-encoder/qnli-distilroberta-base")

        # 평가 대상 모델
        self.model = AsyncOpenAI(base_url=env.SPAM_MODEL_URL, api_key=env.SPAM_MODEL_API_KEY)
        self.model_name = model_name

        # LLM Judge (SOTA 모델 - GPT-4o)
        self.judge_model = AsyncOpenAI(api_key=env.OPENAI_API_KEY)
        self.judge_model_name = "gpt-4o"

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

    def _calculate_semantic_similarity(self, reference: str, prediction: str) -> float:
        """
        CrossEncoder를 사용한 의미론적 유사성 계산
        설명(reference)이 생성된 설명(prediction)과 얼마나 일치하는지 측정
        """
        try:
            scores = self.semantic_model.predict([[reference, prediction]])
            # 점수를 0-1 범위로 정규화
            similarity = float((scores[0] + 1) / 2)  # -1~1 → 0~1
            return min(max(similarity, 0.0), 1.0)
        except Exception as e:
            logger.warning("의미론적 유사성 계산 실패: %s", e)
            return 0.0

    @async_decorator_log(level=logging.DEBUG)
    @async_log_performance
    async def _llm_judge_evaluation(
        self, reference: str, prediction: str, spam_text: str
    ) -> tuple[float, str]:
        """
        LLM as Judge: GPT-4o가 XAI 설명의 품질을 평가
        - 정확성: 설명이 스팸 판정과 일치하는가?
        - 완전성: 모든 근거를 포함했는가?
        - 명확성: 설명이 명확한가?
        """
        judge_prompt = f"""당신은 스팸 탐지 설명 품질 평가 전문가입니다.

[스팸 텍스트]
{spam_text}

[기준 설명 (레퍼런스)]
{reference}

[생성된 설명 (평가 대상)]
{prediction}

위 생성된 설명이 얼마나 좋은지 평가해주세요.

평가 기준:
- 정확성 (Accuracy): 설명이 스팸 판정의 실제 이유와 일치하는가?
- 완전성 (Completeness): 모든 스팸 판정 근거를 포함했는가?
- 명확성 (Clarity): 설명이 명확하고 이해하기 쉬운가?
- 구체성 (Specificity): 구체적인 근거를 제시했는가?

응답 형식:
{{
  "score": <0.0~1.0 사이의 점수>,
  "explanation": "<평가 이유를 한 문장으로>"
}}"""

        try:
            resp = await self.judge_model.chat.completions.create(
                model=self.judge_model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
            )
            
            response_text = resp.choices[0].message.content.strip()
            
            # JSON 파싱
            try:
                result = json.loads(response_text)
                score = float(result.get("score", 0.0))
                explanation = str(result.get("explanation", "평가 실패"))
                return min(max(score, 0.0), 1.0), explanation
            except json.JSONDecodeError:
                logger.warning("LLM Judge 응답 파싱 실패: %s", response_text)
                return 0.0, "JSON 파싱 실패"

        except Exception as e:
            logger.warning("LLM Judge 평가 실패: %s", e)
            return 0.0, f"평가 중 오류: {str(e)}"

    @async_decorator_log(level=logging.DEBUG)
    @async_log_performance
    async def evaluate(self, verbose: bool = False):
        """
        전체 평가 실행: BLEU + Semantic Similarity + LLM as Judge
        """
        metrics_list: list[EvaluationMetrics] = []
        bleu_scores: list[float] = []
        semantic_scores: list[float] = []
        llm_judge_scores: list[float] = []

        for i, sample in enumerate(self.dataset):
            try:
                prompt, reference = self._build_prompt(sample)
                spam_text = sample.get("input", "")

                # 모델 추론
                resp = await self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                prediction = resp.choices[0].message.content.strip()

                # 1. BLEU 점수 계산
                bleu = self.bleu.score(output=prediction, references=[reference])
                bleu_score = float(bleu)
                bleu_scores.append(bleu_score)

                # 2. Semantic Similarity 계산
                semantic_sim = self._calculate_semantic_similarity(reference, prediction)
                semantic_scores.append(semantic_sim)

                # 3. LLM as Judge 평가
                judge_score, judge_explanation = await self._llm_judge_evaluation(
                    reference, prediction, spam_text
                )
                llm_judge_scores.append(judge_score)

                # 메트릭 저장
                metrics = EvaluationMetrics(
                    sample_id=i,
                    prompt=prompt,
                    reference=reference,
                    prediction=prediction,
                    bleu_score=bleu_score,
                    semantic_similarity=semantic_sim,
                    llm_judge_score=judge_score,
                    llm_judge_explanation=judge_explanation,
                )
                metrics_list.append(metrics)

                if verbose and i % 10 == 0:
                    logger.info(
                        f"[{i}/{len(self.dataset)}] BLEU: {bleu_score:.4f}, "
                        f"Semantic: {semantic_sim:.4f}, Judge: {judge_score:.4f}"
                    )

            except Exception as e:
                logger.warning("샘플 %d 처리 스킵: %s", i, e)

        if not bleu_scores:
            raise RuntimeError("유효한 샘플이 없어 점수를 계산할 수 없습니다.")

        # 평가 결과 계산
        results = {
            "bleu": safe_mean(bleu_scores),
            "semantic_similarity": safe_mean(semantic_scores),
            "llm_judge": safe_mean(llm_judge_scores),
            "metrics_details": metrics_list,
            "total_samples": len(metrics_list),
        }

        return results

    def print_leaderboard(self, results: dict):
        """리더보드 형태로 결과 출력"""
        print("\n" + "=" * 80)
        print("🏆 SPAM XAI EVALUATION LEADERBOARD 🏆".center(80))
        print("=" * 80 + "\n")

        # 종합 점수 계산 (가중 평균)
        weights = {"bleu": 0.25, "semantic_similarity": 0.35, "llm_judge": 0.40}
        overall_score = (
            results["bleu"] * weights["bleu"]
            + results["semantic_similarity"] * weights["semantic_similarity"]
            + results["llm_judge"] * weights["llm_judge"]
        )

        # 메인 점수 표시
        print(f"{'Model':<20} {'BLEU':<12} {'Semantic':<12} {'LLM Judge':<12} {'Overall':<12}")
        print("-" * 68)
        print(
            f"{self.model_name:<20} {results['bleu']:<12.4f} "
            f"{results['semantic_similarity']:<12.4f} {results['llm_judge']:<12.4f} "
            f"{overall_score:<12.4f}"
        )

        print("\n" + "-" * 80)
        print("📊 Metric Details".center(80))
        print("-" * 80)

        print(f"\n✓ BLEU Score (어휘 일치도): {results['bleu']:.4f}")
        print(f"  - 범위: 0~1 (높을수록 좋음)")
        print(f"  - 설명이 기준 설명과 얼마나 일치하는지 측정\n")

        print(f"✓ Semantic Similarity (의미론적 유사성): {results['semantic_similarity']:.4f}")
        print(f"  - 범위: 0~1 (높을수록 좋음)")
        print(f"  - 생성된 설명의 의미가 기준과 일치하는 정도\n")

        print(f"✓ LLM Judge Score (설명 품질): {results['llm_judge']:.4f}")
        print(f"  - 범위: 0~1 (높을수록 좋음)")
        print(f"  - 정확성, 완전성, 명확성, 구체성을 GPT-4o가 평가\n")

        print(f"✓ Overall Score (종합 점수): {overall_score:.4f}")
        print(f"  - BLEU 25% + Semantic 35% + LLM Judge 40%\n")

        print("=" * 80)
        print(f"평가된 샘플 수: {results['total_samples']}")
        print("=" * 80 + "\n")

    def save_detailed_results(self, results: dict, output_path: str = "./eval/results.json"):
        """상세 평가 결과 저장"""
        export_data = {
            "summary": {
                "bleu": results["bleu"],
                "semantic_similarity": results["semantic_similarity"],
                "llm_judge": results["llm_judge"],
                "total_samples": results["total_samples"],
            },
            "details": [
                {
                    "sample_id": m.sample_id,
                    "reference": m.reference,
                    "prediction": m.prediction,
                    "bleu_score": m.bleu_score,
                    "semantic_similarity": m.semantic_similarity,
                    "llm_judge_score": m.llm_judge_score,
                    "llm_judge_explanation": m.llm_judge_explanation,
                }
                for m in results["metrics_details"]
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"상세 결과 저장: {output_path}")


if __name__ == "__main__":
    import sys
    import argparse
    
    async def main():
        """평가 실행을 위한 메인 함수"""
        # CLI 인수 파싱
        parser = argparse.ArgumentParser(description="LLM XAI 평가 스크립트")
        parser.add_argument("--limit", type=int, default=None, help="평가할 샘플 수 제한")
        parser.add_argument("--verbose", type=lambda x: x.lower() == "true", default=False, help="상세 로그 출력")
        parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="평가할 모델명")
        
        args = parser.parse_args()
        
        logger.info("🚀 LLM XAI 평가를 시작합니다...")
        logger.info(f"📊 설정: limit={args.limit}, verbose={args.verbose}, model={args.model}")

        # 평가 인스턴스 생성
        evaluator = LLMEvaluation(limit=args.limit, model_name=args.model)

        try:
            results = await evaluator.evaluate(verbose=args.verbose)

            # 리더보드 출력
            evaluator.print_leaderboard(results)

            # 상세 결과 저장
            evaluator.save_detailed_results(results)

        except Exception as e:
            logger.error("평가 중 오류 발생: %s", e)
            raise

    # 비동기 메인 함수 실행
    asyncio.run(main())