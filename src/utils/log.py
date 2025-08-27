import logging
import sys
import functools
from datetime import datetime

# 로거 설정
logger = logging.getLogger('skitty')
logger.setLevel(logging.DEBUG) # 로그레벨 설정
logger.propagate = False # 중복로그 방지

# 핸들러가 없는 경우에만 추가
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout) # 콘솔에서 로그 출력
    handler.setLevel(logging.DEBUG)

    # 로거이름 %(name)s::, # 로그 시간 %(asctime)::s, 로그 레벨 %(levelname)::s
    # 로그가 찍힌 파이썬 py %(modeul)s, 로그 찍은 함수 %(funcName)s
    # 실제 찍힌 로그 메세지 %(message)s
    formatter = logging.Formatter(
        " %(name)s::%(asctime)s::%(levelname)s::[%(module)s.%(funcName)s]::\n%(message)s"
        , datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 데코레이터 패턴 적용
def decorator_log(level=logging.DEBUG):
    '''데코레이터 패턴 적용'''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, f"CALL {func.__name__}() args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"RETURN {func.__name__} -> {result}")
                return result
            except Exception as e:
                logger.log(level, f"EXCEPTION {func.__name__} -> {e}")
                raise e
        return wrapper
    return decorator

# 비동기 함수용 데코레이터 패턴 추가
def async_decorator_log(level=logging.DEBUG):
    '''비동기 함수용 데코레이터 패턴 추가'''
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger.log(level, f"CALL {func.__name__}() args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                if hasattr(result, '__aiter__'):
                    logger.log(level, f"RETURN {func.__name__} -> <AsyncGenerator>")
                else:
                    logger.log(level, f"RETURN {func.__name__} -> {result}")
                return result
            except Exception as e:
                logger.log(level, f"EXCEPTION {func.__name__} -> {e}")
                raise e
        return wrapper
    return decorator
    
# 성능 측정 데코레이터
def log_performance(func):
    """함수 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"PERFORMANCE {func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(f"PERFORMANCE {func.__name__} failed after {execution_time:.4f}s")
            raise
    return wrapper

# 비동기 성능 측정 데코레이터
def async_log_performance(func):
    """비동기 함수 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"PERFORMANCE {func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(f"PERFORMANCE {func.__name__} failed after {execution_time:.4f}s")
            raise
    return wrapper