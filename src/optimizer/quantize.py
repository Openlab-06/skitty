"""
모델 양자화 스크립트 (AWQ / GPTQ)
학습된 모델을 AWQ 또는 GPTQ로 양자화하여 추론 속도 향상 및 메모리 사용량 감소
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.log import logger


class QuantizationConfig:
    """양자화 설정"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        quantization_type: str = "awq",
        bits: int = 4,
        group_size: int = 128,
        calibration_samples: int = 512,
        dataset_name: str = "Devocean-06/Spam_QA-Corpus",
        dataset_split: str = "train",
        max_seq_length: int = 1500,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.quantization_type = quantization_type.lower()
        self.bits = bits
        self.group_size = group_size
        self.calibration_samples = calibration_samples
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.max_seq_length = max_seq_length
        
        # 양자화 타입 검증
        if self.quantization_type not in ["awq", "gptq"]:
            raise ValueError(f"지원하지 않는 양자화 타입입니다: {quantization_type}")
        
        # 출력 디렉토리 생성
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class CalibrationDataProcessor:
    """Calibration 데이터 전처리기"""
    
    def __init__(self, tokenizer, config: QuantizationConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def _build_prompt(self, sample: dict) -> str:
        """Alpaca 포맷(instruction/input/output)을 프롬프트로 변환"""
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        
        if input_text:
            # Chat 템플릿 형식으로 변환
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        return prompt
    
    def prepare_calibration_data(self) -> List[str]:
        """
        Calibration 데이터 준비
        - Alpaca 형식 데이터셋을 로드하여 텍스트로 변환
        """
        logger.info(f"📊 Calibration 데이터 로드: {self.config.dataset_name}")
        
        # 데이터셋 로드
        dataset = load_dataset(
            self.config.dataset_name, 
            split=self.config.dataset_split
        )
        
        # 샘플 수 제한
        if len(dataset) > self.config.calibration_samples:
            dataset = dataset.shuffle(seed=42).select(range(self.config.calibration_samples))
        
        # 텍스트 변환
        calibration_texts = []
        for sample in dataset:
            prompt = self._build_prompt(sample)
            
            # Tokenizer의 chat template이 있으면 사용
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                text = prompt
            
            calibration_texts.append(text)
        
        logger.info(f"✅ Calibration 데이터 준비 완료: {len(calibration_texts)}개 샘플")
        return calibration_texts
    
    def prepare_calibration_tokens(self) -> List[torch.Tensor]:
        """
        Calibration 데이터를 토크나이즈
        """
        texts = self.prepare_calibration_data()
        
        logger.info(f"🔄 Calibration 데이터 토크나이즈 중...")
        tokenized_data = []
        
        for text in texts:
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding=False,
            )
            tokenized_data.append(tokens["input_ids"])
        
        logger.info(f"✅ 토크나이즈 완료: {len(tokenized_data)}개 샘플")
        return tokenized_data


class AWQQuantizer:
    """AWQ 양자화기"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
    
    def quantize(self):
        """AWQ 양자화 수행"""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "AWQ 라이브러리가 설치되어 있지 않습니다. "
                "다음 명령어로 설치하세요: pip install autoawq"
            )
        
        logger.info("🚀 AWQ 양자화 시작")
        logger.info(f"   - 모델: {self.config.model_path}")
        logger.info(f"   - Bits: {self.config.bits}")
        logger.info(f"   - Group Size: {self.config.group_size}")
        
        # 토크나이저 로드
        logger.info("📚 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Calibration 데이터 준비
        data_processor = CalibrationDataProcessor(tokenizer, self.config)
        calibration_texts = data_processor.prepare_calibration_data()
        
        # AWQ 모델 로드
        logger.info("🔧 AWQ 모델 로드 중...")
        model = AutoAWQForCausalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
        )
        
        # 양자화 설정
        quant_config = {
            "zero_point": True,
            "q_group_size": self.config.group_size,
            "w_bit": self.config.bits,
            "version": "GEMM"
        }
        
        # 양자화 수행
        logger.info("⚡ AWQ 양자화 수행 중... (시간이 걸릴 수 있습니다)")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_texts,
        )
        
        # 양자화된 모델 저장
        logger.info(f"💾 양자화된 모델 저장: {self.config.output_dir}")
        model.save_quantized(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ AWQ 양자화 완료!")


class GPTQQuantizer:
    """GPTQ 양자화기"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
    
    def quantize(self):
        """GPTQ 양자화 수행"""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "GPTQ 라이브러리가 설치되어 있지 않습니다. "
                "다음 명령어로 설치하세요: pip install auto-gptq"
            )
        
        logger.info("🚀 GPTQ 양자화 시작")
        logger.info(f"   - 모델: {self.config.model_path}")
        logger.info(f"   - Bits: {self.config.bits}")
        logger.info(f"   - Group Size: {self.config.group_size}")
        
        # 토크나이저 로드
        logger.info("📚 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=True,
        )
        
        # Calibration 데이터 준비
        data_processor = CalibrationDataProcessor(tokenizer, self.config)
        calibration_texts = data_processor.prepare_calibration_data()
        
        # GPTQ 양자화 설정
        quantize_config = BaseQuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=False,  # activation에 대한 정렬 비활성화 (속도 향상)
            damp_percent=0.01,
        )
        
        # GPTQ 모델 로드
        logger.info("🔧 GPTQ 모델 로드 중...")
        model = AutoGPTQForCausalLM.from_pretrained(
            self.config.model_path,
            quantize_config=quantize_config,
            device_map="auto",
        )
        
        # 양자화용 데이터셋 준비
        logger.info("📊 Calibration 데이터셋 구성 중...")
        
        # GPTQ는 리스트의 딕셔너리 형태를 요구
        calibration_dataset = []
        for text in calibration_texts[:self.config.calibration_samples]:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
            )
            calibration_dataset.append(tokens)
        
        # 양자화 수행
        logger.info("⚡ GPTQ 양자화 수행 중... (시간이 걸릴 수 있습니다)")
        model.quantize(
            calibration_dataset,
            batch_size=1,
        )
        
        # 양자화된 모델 저장
        logger.info(f"💾 양자화된 모델 저장: {self.config.output_dir}")
        model.save_quantized(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ GPTQ 양자화 완료!")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="모델 양자화 (AWQ / GPTQ)")
    
    # 필수 인자
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="양자화할 모델 경로 (merged 모델 또는 base 모델)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="양자화된 모델 저장 경로"
    )
    
    # 양자화 설정
    parser.add_argument(
        "--quantization-type",
        type=str,
        default="awq",
        choices=["awq", "gptq"],
        help="양자화 타입 (기본값: awq)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="양자화 비트 수 (기본값: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="그룹 사이즈 (기본값: 128)"
    )
    
    # Calibration 데이터 설정
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=512,
        help="Calibration 샘플 수 (기본값: 512)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Devocean-06/Spam_QA-Corpus",
        help="Calibration 데이터셋 이름"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="데이터셋 스플릿 (기본값: train)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1500,
        help="최대 시퀀스 길이 (기본값: 1500)"
    )
    
    args = parser.parse_args()
    
    # 양자화 설정 생성
    config = QuantizationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization_type=args.quantization_type,
        bits=args.bits,
        group_size=args.group_size,
        calibration_samples=args.calibration_samples,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_seq_length=args.max_seq_length,
    )
    
    # 양자화 수행
    try:
        if config.quantization_type == "awq":
            quantizer = AWQQuantizer(config)
        else:  # gptq
            quantizer = GPTQQuantizer(config)
        
        quantizer.quantize()
        
        logger.info("=" * 80)
        logger.info("🎉 양자화 완료!")
        logger.info(f"   - 양자화 타입: {config.quantization_type.upper()}")
        logger.info(f"   - 저장 경로: {config.output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ 양자화 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

