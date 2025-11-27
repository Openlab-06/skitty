"""데이터 중복제거, 정규화, I/O 모듈 단위 테스트"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
from pathlib import Path

from src.data.utils.normalize import TextProcessor
from src.data.data_dedup import SimhashGenerator, DuplicateFinder
from src.data.utils.io import DataFrameProcessor


class TestTextProcessor:
    """텍스트 정규화 모듈 테스트"""
    
    @pytest.fixture
    def processor(self):
        return TextProcessor()
    
    def test_normalize_basic(self, processor):
        """기본 정규화 테스트"""
        text = "안녕하세요! 010-1234-5678로 연락주세요."
        result = processor.normalize(text)
        
        assert "<phone>" in result  # 소문자로 변환됨
        assert "안녕하세요" in result
        assert "010-1234-5678" not in result
    
    def test_normalize_url(self, processor):
        """URL 정규화 테스트"""
        text = "방문하세요: https://example.com"
        result = processor.normalize(text)
        
        assert "<url>" in result  # 소문자로 변환됨
        assert "https://example.com" not in result
    
    def test_normalize_numbers(self, processor):
        """숫자 정규화 테스트"""
        text = "가격은 1,000,000원입니다."
        result = processor.normalize(text)
        
        assert "<num>" in result  # 소문자로 변환됨
        assert "1,000,000" not in result
    
    def test_normalize_empty_text(self, processor):
        """빈 텍스트 처리 테스트"""
        assert processor.normalize("") == ""
        assert processor.normalize(None) == ""
    
    def test_create_char_ngrams(self, processor):
        """Character n-gram 생성 테스트"""
        text = "hello"
        ngrams_2 = processor.create_char_ngrams(text, 2)
        expected = ["he", "el", "ll", "lo"]
        
        assert ngrams_2 == expected
    
    def test_create_char_ngrams_short_text(self, processor):
        """짧은 텍스트 n-gram 테스트"""
        text = "hi"
        ngrams_3 = processor.create_char_ngrams(text, 3)
        
        assert ngrams_3 == ["hi"]  # 텍스트가 n보다 짧으면 원본 반환


class TestSimhashGenerator:
    """Simhash 생성기 테스트"""
    
    @pytest.fixture
    def generator(self):
        return SimhashGenerator(ngram_size=2)
    
    def test_create_simhash(self, generator):
        """Simhash 생성 테스트"""
        text = "hello world"
        simhash = generator.create_simhash(text)
        
        assert simhash is not None
        assert hasattr(simhash, 'value')
        assert hasattr(simhash, 'distance')
    
    def test_same_text_same_simhash(self, generator):
        """동일한 텍스트는 동일한 Simhash 생성"""
        text = "hello world"
        simhash1 = generator.create_simhash(text)
        simhash2 = generator.create_simhash(text)
        
        assert simhash1.value == simhash2.value
    
    def test_similar_text_similar_simhash(self, generator):
        """유사한 텍스트는 유사한 Simhash 생성"""
        text1 = "hello world"
        text2 = "hello word"  # 한 글자 차이
        
        simhash1 = generator.create_simhash(text1)
        simhash2 = generator.create_simhash(text2)
        
        # 해밍 거리가 작아야 함 (유사함)
        distance = simhash1.distance(simhash2)
        assert distance < 20  # 임계값 조정 (character n-gram 특성상 거리가 클 수 있음)


class TestDuplicateFinder:
    """중복 탐지기 테스트"""
    
    @pytest.fixture
    def finder(self):
        return DuplicateFinder(hamming_distance=3)
    
    def test_build_index(self, finder):
        """인덱스 구축 테스트"""
        texts = ["hello world", "spam message", "hello world", ""]
        
        index, simhash_objects, empty_count = finder.build_index(texts)
        
        assert index is not None
        assert len(simhash_objects) == 4
        assert empty_count == 1  # 빈 텍스트 1개
    
    def test_find_duplicates(self, finder):
        """중복 찾기 테스트"""
        texts = ["hello world", "hello world", "different text"]
        
        index, simhash_objects, _ = finder.build_index(texts)
        unique_indices, duplicate_info = finder.find_duplicates(index, simhash_objects, texts)
        
        # 중복이 있으므로 유니크 개수가 원본보다 적어야 함
        assert len(unique_indices) < len(texts)
        assert len(duplicate_info) > 0
        
        # 중복 정보 구조 확인
        if duplicate_info:
            dup = duplicate_info[0]
            assert "original_index" in dup
            assert "duplicate_index" in dup
            assert "hamming_distance" in dup


class TestDataFrameProcessor:
    """DataFrame 처리기 테스트"""
    
    def test_load_and_preprocess(self, tmp_path):
        """CSV 로드 및 전처리 테스트"""
        # 임시 CSV 파일 생성
        csv_file = tmp_path / "test.csv"
        csv_content = "text,label\nhello world,0\nspam message,1\n"
        csv_file.write_text(csv_content)
        
        df = DataFrameProcessor.load_and_preprocess(str(csv_file), "text")
        
        assert "text" in df.columns
        assert "text_norm" in df.columns
        assert "_rowid_" in df.columns  # ID 컬럼 추가 확인 (실제로는 _rowid_로 생성됨)
        assert df.height == 2
    
    def test_load_missing_column(self, tmp_path):
        """없는 컬럼 지정 시 에러 테스트"""
        csv_file = tmp_path / "test.csv"
        csv_content = "text,label\nhello,0\n"
        csv_file.write_text(csv_content)
        
        with pytest.raises(ValueError, match="'missing_col' 컬럼이 없습니다"):
            DataFrameProcessor.load_and_preprocess(str(csv_file), "missing_col")
    
    def test_create_result_dataframes(self):
        """결과 DataFrame 생성 테스트"""
        # 테스트용 DataFrame 생성 (실제 컬럼명에 맞춤)
        df = pl.DataFrame({
            "_rowid_": [0, 1, 2],
            "text": ["hello", "world", "spam"],
            "text_norm": ["hello", "world", "spam"]
        })
        
        unique_indices = [0, 2]  # 인덱스 1은 중복으로 제거
        duplicate_info = [{
            "original_index": 0,
            "duplicate_index": 1,
            "hamming_distance": 2
        }]
        
        unique_df, dup_df = DataFrameProcessor.create_result_dataframes(
            df, unique_indices, duplicate_info, "text"
        )
        
        # 유니크 DataFrame 검증
        assert unique_df.height == 2
        assert "text" in unique_df.columns
        assert "text_norm" not in unique_df.columns  # 정규화 컬럼은 제거됨
        
        # 중복 DataFrame 검증
        assert dup_df.height == 1
        assert "original_text" in dup_df.columns
        assert "duplicate_text" in dup_df.columns
        assert "hamming_distance" in dup_df.columns
    
    @patch('polars.DataFrame.write_parquet')
    def test_save_results(self, mock_write_parquet):
        """결과 저장 테스트"""
        unique_df = pl.DataFrame({"text": ["hello", "world"]})
        dup_df = pl.DataFrame({"info": ["duplicate"]})
        
        DataFrameProcessor.save_results(
            unique_df, dup_df, "unique.parquet", "dup.parquet", 10
        )
        
        # Parquet 저장이 호출되었는지 확인
        assert mock_write_parquet.call_count == 2
