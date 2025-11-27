from enum import Enum

class DataProcessingError(Enum):
    """데이터 처리 에러 코드 및 메시지"""
    DATA_DEDUPLICATION_ERROR = "DATA_DEDUPLICATION_ERROR"
    DATA_FILTERING_ERROR = "DATA_FILTERING_ERROR"
    DATA_AUGMENTATION_ERROR = "DATA_AUGMENTATION_ERROR"

class BaseDataProcessingException(Exception):
    """데이터 처리 공통 예외"""
    def __init__(self, error: DataProcessingError, detail: str = None, status_code: int = 500):
        self.error = error
        self.status_code = status_code
        super().__init__(f"[{self.error}] {self.error}: {detail}" if detail else f"[{self.error}] {self.error}")


class DataDeduplicationError(BaseDataProcessingException):
    """데이터 중복제거 예외"""
    def __init__(self, error: DataProcessingError.DATA_DEDUPLICATION_ERROR, status_code: int = 422):
        super().__init__(error, status_code=status_code)


class DataFilteringError(BaseDataProcessingException):
    """데이터 필터링 예외"""
    def __init__(self, error: DataProcessingError.DATA_FILTERING_ERROR, status_code: int = 400):
        super().__init__(error, status_code=status_code)


class DataAugmentationError(BaseDataProcessingException):
    """데이터 증강 예외"""
    def __init__(self, error: DataProcessingError.DATA_AUGMENTATION_ERROR, status_code: int = 500):
        super().__init__(error, status_code=status_code)
