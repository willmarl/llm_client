import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import get_llm, run_llm

print("""
Running llm tests of: get_llm(), run_llm()
""")

try:
    response = get_llm().invoke("")
    if response:
        print("get_llm passed ✅")
except Exception as e:
    print(f"get_llm failed ❌: {e}")

try:
    response = run_llm("hello")
    if response:
        print("run_llm passed ✅")
except Exception as e:
    print(f"run_llm failed ❌: {e}")
