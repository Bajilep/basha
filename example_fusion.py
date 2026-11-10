
"""Interactive test for the Basha English Fusion pipeline."""

import json
import time
import traceback
from pathlib import Path

from basha.fusion.english_fusion import EnglishFusion


def display_result(result, elapsed):
    sentiment = result.get("sentiment") or {}
    if isinstance(sentiment, dict):
        label = sentiment.get("label", "unknown")
        confidence = sentiment.get("confidence")
    else:
        label = str(sentiment)
        confidence = None

    print("\n" + "=" * 65)
    print("BASHA — ENGLISH FUSION RESULT")
    print("=" * 65)
    print(f"Final sentiment : {label.upper()}")
    if confidence is not None:
        print(f"Confidence      : {confidence:.4f}")

    print(f"Decision source : {result.get('decision_source', 'unknown')}")
    print(f"Processing time : {elapsed:.2f} seconds")

    core = result.get("core_sentiment") or {}
    print("\nOVERALL SENTIMENT")
    print("-" * 65)
    print(f"Label    : {core.get('label', 'N/A')}")
    for key in ("positive", "neutral", "negative"):
        print(f"{key.title():9}: {core.get(key, 0):.4f}")

    print("\nASPECT-BASED SENTIMENT")
    print("-" * 65)
    aspects = result.get("aspects") or []

    if not aspects:
        print("No aspects detected.")
    else:
        for i, aspect in enumerate(aspects, 1):
            print(
                f"{i}. {aspect.get('aspect', '')} "
                f"-> {str(aspect.get('sentiment', 'unknown')).upper()} "
                f"({aspect.get('confidence', 0):.4f})"
            )

    mixed = result.get("mixed_analysis") or {}
    print("\nMIXED SENTIMENT")
    print("-" * 65)
    print(f"Detected   : {mixed.get('mixed', False)}")
    print(f"Confidence : {mixed.get('confidence', 0)}")
    print(f"Reasons    : {mixed.get('reasons', [])}")

    contrast = result.get("contrast") or {}
    print("\nCONTRAST ANALYSIS")
    print("-" * 65)
    print(f"Detected : {contrast.get('contrast_detected', False)}")
    print(f"Marker   : {contrast.get('marker')}")
    print(f"Label    : {contrast.get('label')}")

    for i, clause in enumerate(contrast.get("clauses") or [], 1):
        print(
            f"  Clause {i}: {clause.get('text', '')}\n"
            f"  Sentiment: {clause.get('label', 'unknown')} "
            f"({clause.get('confidence', 0):.4f})"
        )

    sarcasm = result.get("sarcasm") or {}
    print("\nSARCASM")
    print("-" * 65)
    print(f"Detected : {sarcasm.get('detected', False)}")
    print(f"Score    : {sarcasm.get('sarcasm', 0)}")

    print("\nSENTENCE ANALYSIS")
    print("-" * 65)
    for i, sentence in enumerate(result.get("sentences") or [], 1):
        print(
            f"{i}. {sentence.get('text', '')}\n"
            f"   -> {sentence.get('label', 'unknown')} "
            f"({sentence.get('confidence', 0):.4f})"
        )

    print("\nADDITIONAL INTERNAL DETAILS")
    print("-" * 65)
    for key in ("discourse", "semantic_conflict", "sarcasm_resolution"):
        print(f"\n{key.upper()}:")
        print(json.dumps(result.get(key), indent=2, ensure_ascii=False))


def read_multiline():
    print("\nPaste the complete text.")
    print("Type END on a separate line to analyze.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()


def read_file():
    filename = input("Text file path: ").strip().strip('"')
    return Path(filename).expanduser().read_text(encoding="utf-8").strip()


def main():
    print("=" * 65)
    print("BASHA INTERACTIVE SENTIMENT TEST")
    print("=" * 65)
    print("Loading pretrained models. Please wait...\n")

    analyzer = EnglishFusion()

    print("\nBasha is ready!")

    while True:
        print("\nChoose input mode:")
        print("1. Single-line text")
        print("2. Multiline text")
        print("3. Read a UTF-8 text file")
        print("4. Exit")

        try:
            choice = input("\nChoice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Basha.")
            break

        if choice == "4":
            print("Goodbye!")
            break

        try:
            if choice == "1":
                text = input("Enter text: ").strip()
            elif choice == "2":
                text = read_multiline()
            elif choice == "3":
                text = read_file()
            else:
                print("Invalid choice.")
                continue

            if not text:
                print("Please enter some text.")
                continue

            print(f"\nCharacters received : {len(text)}")
            print(f"Approximate words    : {len(text.split())}")

            start = time.perf_counter()
            result = analyzer.analyze(text)
            elapsed = time.perf_counter() - start

            display_result(result, elapsed)

        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
        except Exception:
            print("\nAnalysis failed:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
