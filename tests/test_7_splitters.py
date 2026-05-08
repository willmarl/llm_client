from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src import load_any_file, splitter
from config import FILE_PATH

print(
    """
Running splitter test of: load_any_file()
"""
)

file_location = str(FILE_PATH)
file_path = FILE_PATH

if not file_path.exists():
    print("file not found. halting tests")
    sys.exit(1)  # or just return if inside a function

try:
    doc = load_any_file(file_location)
    response = splitter(doc)
    if response:
        print("load_any_file passed ✅")
except Exception as e:
    print(f"load_any_file failed ❌: {e}")
