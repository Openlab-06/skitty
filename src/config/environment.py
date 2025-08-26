from functools import lru_cache
from src.utils.log import logger
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path
import os
from typing import Optional

@lru_cache
def get_root_dir():
    '''프로젝트 root 경로 반환'''
    return Path(__file__).parent.parent.parent

@lru_cache
def get_env_file():
    '''환경변수 파일 반환
    ENV : local, prd, stg
    '''
    runtime_env = os.getenv("ENV","")
    logger.info(f"Runtime environment: {runtime_env}")
    return f"{runtime_env}.env" if runtime_env else ".env"

class Environment(BaseSettings):
    '''rag 프로젝트 환경변수 설정'''
    # gemini 설정
    GEMINI_API_KEY : str
    GEMINI_MODEL_ID : str

    model_config = ConfigDict(
        env_file = get_env_file(),
        env_file_encoding='utf-8', 
        extra="ignore", 
        validate_assignment=True 
    )

@lru_cache
def get_environment_variables():
    logger.info(f"Environment Setting::get_environment_variables:: [환경설정] get_environment_variables() 시작")
    try:
        settings = Environment()
        logger.info(f"Environment Setting::get_environment_variables:: [환경설정] 환경변수 로딩 성공")
        return settings
    except Exception as e:
        import traceback
        trace = traceback.format_exc() 
        logger.info(f"Environment Setting::get_environment_variables:: [환경설정] 환경변수 로딩 실패 {trace}")
        raise