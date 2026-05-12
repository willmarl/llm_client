from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src import load_raw_text, load_any_file, load_folder
from config import FILE_PATH, FOLDER_PATH

print(
    """
Running loader tests of: load_raw_text(), load_any_file(), load_folder()
"""
)

file_location = str(FILE_PATH)
file_path = FILE_PATH
folder_location = str(FOLDER_PATH)
folder_path = FOLDER_PATH

if not file_path.exists():
    print("file not found. halting tests")
    sys.exit(1)  # or just return if inside a function
if not folder_path.exists():
    print("folder not found. halting tests")
    sys.exit(1)  # or just return if inside a function

try:
    response = load_raw_text("hi")
    if response:
        print("load_raw_text passed ✅")
except Exception as e:
    print(f"load_raw_text failed ❌: {e}")

try:
    response = load_any_file(file_location)
    if isinstance(response, list) and len(response) > 0:
        print("load_any_file passed ✅")
    else:
        print("load_any_file failed ❌: returned empty or wrong type")
except Exception as e:
    print(f"load_any_file failed ❌: {e}")

try:
    response = load_folder(folder_location)
    if response:
        print("load_folder passed ✅")
except Exception as e:
    print(f"load_folder failed ❌: {e}")
