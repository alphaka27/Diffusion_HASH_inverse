"""
공용 유틸리티
- file_io: 파일 입출력/레코드 포맷
"""

# 코드베이스에 따라 클래스명이 FileIO 또는 FILEio 일 수 있어 호환 처리
try:
    from .project_root import add_root_to_path, add_src_to_path  # 프로젝트 루트 경로 설정 유틸
    from .file_io import FileIO  # 권장 표기
    from .json_formatter import OutputFormat
    from .csv_formatter import CSVFormat
except ImportError:  # 기존 코드 호환
    from .file_io import FILEio as FileIO

__all__ = [
    "FileIO",
    "add_root_to_path",
    "add_src_to_path",
    "OutputFormat",
    "CSVFormat"
]
