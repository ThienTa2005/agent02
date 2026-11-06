"""Giúp pytest import được package `arena` khi chạy từ thư mục gốc."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
