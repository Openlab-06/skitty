from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from src.utils.log import logger

@lru_cache
def get_project_root():
    """프로젝트 루트 디렉토리 반환"""
    return Path(__file__).parent.parent.parent

class ProjectConfig(BaseSettings):
    """프로젝트 환경변수 설정"""
    # Gemini API 설정
    GEMINI_API_KEY: str
    GEMINI_MODEL_ARGU: str
    GEMINI_MODEL_FILTER: str

    # spam model
    SPAM_MODEL: str
    SPAM_MODEL_URL: str
    SPAM_MODEL_API_KEY: str

    # OpenAI API 설정 (페일오버용, 선택적)
    OPENAI_API_KEY: str = None  # 없으면 페일오버 비활성화
    OPENAI_MODEL: str = None
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_config():
    """환경설정 싱글톤 반환"""
    try:
        return ProjectConfig()
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        raise e