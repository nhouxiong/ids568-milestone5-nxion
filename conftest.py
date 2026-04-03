"""Pytest configuration — ensures src/ is importable as a package."""
import sys
from pathlib import Path

# Add project root to sys.path so `from src.xxx import ...` works
sys.path.insert(0, str(Path(__file__).parent))
