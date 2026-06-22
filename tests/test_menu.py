import subprocess
import sys
import pytest

pytest.importorskip("questionary", reason="install llm_client[full] to run the interactive test menu")

import questionary
from questionary import Choice

TESTS = [
    {
        "name": "test_1_llm.py",
        "label": "1 · LLM              — get_llm(), run_llm()",
    },
    {
        "name": "test_2_image.py",
        "label": "2 · Image LLM        — get_image_llm(), generate_image_caption(), generate_image_ocr_text()",
    },
    {
        "name": "test_3_text-embed.py",
        "label": "3 · Text Embeddings  — get_text_embeddings(), generate_text_embeddings()",
    },
    {
        "name": "test_4_image-embed.py",
        "label": "4 · Image Embeddings — generate_image_embeddings()",
    },
    {
        "name": "test_5_ultimate-image-extractor.py",
        "label": "5 · Ultimate Image   — ultimate_image_extractor()",
    },
    {
        "name": "test_6_loaders.py",
        "label": "6 · Loaders          — load_raw_text(), load_any_file(), load_folder()",
    },
    {
        "name": "test_7_splitters.py",
        "label": "7 · Splitters        — splitter()",
    },
    {
        "name": "test_8_chroma.py",
        "label": "8 · Chroma           — chroma client",
    },
]


def main():
    print(
        """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ \033[1m⚠  REMINDER\033[0m                                  ┃
┃                                                  ┃
┃ These tests are based off your current env.    ┃
┃ If they pass with ollama but you switch to     ┃
┃ openai, they may not pass again.               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""
    )
    print()
    selected_labels = questionary.checkbox(
        "Select tests to run  (Space to toggle, Enter to confirm)",
        choices=[
            Choice(title=t["label"], value=t["label"], checked=True) for t in TESTS
        ],
        instruction="(all selected by default)",
    ).ask()

    if selected_labels is None:
        # user hit Ctrl-C
        print("\nAborted.")
        sys.exit(0)

    if not selected_labels:
        print("\nNo tests selected — nothing to run.")
        sys.exit(0)

    label_to_file = {t["label"]: t["name"] for t in TESTS}
    to_run = [label_to_file[label] for label in selected_labels]

    print(f"\nRunning {len(to_run)} test(s)…\n")
    divider = "─" * 60

    results = []  # list of (filename, passed: bool)
    for filename in to_run:
        print(divider)
        print(f"▶  {filename}")
        print(divider)
        result = subprocess.run([sys.executable, f"tests/{filename}"])
        passed = result.returncode == 0
        results.append((filename, passed))
        print()

    # ── Summary table ────────────────────────────────────────
    col_w = max(len(f) for f, _ in results)
    col_w = max(col_w, 20)
    status_w = 6  # "PASSED" / "FAILED"

    top = f"┌{'─' * (col_w + 2)}┬{'─' * (status_w + 2)}┐"
    mid = f"├{'─' * (col_w + 2)}┼{'─' * (status_w + 2)}┤"
    bottom = f"└{'─' * (col_w + 2)}┴{'─' * (status_w + 2)}┘"

    print(top)
    for filename, passed in results:
        label = "PASSED" if passed else "FAILED"
        icon = "✅" if passed else "❌"
        print(f"│ {filename:<{col_w}} │ {icon} {label} │")
    print(mid)

    n_passed = sum(1 for _, p in results if p)
    n_failed = len(results) - n_passed
    parts = []
    if n_passed:
        parts.append(f"{n_passed} passed")
    if n_failed:
        parts.append(f"{n_failed} failed")
    summary_text = ", ".join(parts)
    print(f"│ {summary_text:<{col_w}} │ {'':>{status_w + 1}} │")
    print(bottom)


if __name__ == "__main__":
    main()
