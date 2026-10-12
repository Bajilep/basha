from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MAMS_DIR = (
    BASE_DIR
    / "data"
    / "MAMS-ATSA"
    / "raw"
)

MAMS_FILES = {
    "train": MAMS_DIR / "train.xml",
    "validation": MAMS_DIR / "val.xml",
    "test": MAMS_DIR / "test.xml",
}

FROZEN_BENCHMARK = (
    BASE_DIR
    / "data"
    / "english_fusion_benchmark.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "mams_dataset_inspection.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Used ONLY for duplicate/leakage checking.

    We preserve the original text everywhere else.
    """
    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


def normalize_polarity(
    polarity: str | None,
) -> str | None:

    if polarity is None:
        return None

    value = polarity.strip().lower()

    mapping = {
        "positive": "positive",
        "pos": "positive",

        "negative": "negative",
        "neg": "negative",

        "neutral": "neutral",
        "neu": "neutral",

        "conflict": "conflict",
    }

    return mapping.get(
        value,
        value,
    )


# ============================================================
# CHECK FILES
# ============================================================

def check_files() -> None:

    print("\nChecking MAMS files...")
    print("-" * 60)

    missing = []

    for split_name, path in MAMS_FILES.items():

        if path.exists():
            print(
                f"✓ {split_name:<10} "
                f"{path}"
            )
        else:
            print(
                f"✗ {split_name:<10} "
                f"{path}"
            )

            missing.append(path)

    if missing:

        print(
            "\nERROR: Some MAMS files "
            "are missing."
        )

        print(
            "\nExpected folder:"
        )

        print(MAMS_DIR)

        raise FileNotFoundError(
            "MAMS XML files not found."
        )


# ============================================================
# LOAD FROZEN BASHA BENCHMARK
# ============================================================

def load_frozen_texts() -> set[str]:

    if not FROZEN_BENCHMARK.exists():

        raise FileNotFoundError(
            f"Frozen benchmark not found:\n"
            f"{FROZEN_BENCHMARK}"
        )

    with FROZEN_BENCHMARK.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, dict):

        if "examples" in data:
            data = data["examples"]

        elif "data" in data:
            data = data["data"]

    if not isinstance(data, list):

        raise TypeError(
            "Frozen benchmark must "
            "contain a list of examples."
        )

    frozen = set()

    for item in data:

        text = item.get("text")

        if text:

            frozen.add(
                normalize_text(text)
            )

    return frozen


# ============================================================
# XML HELPERS
# ============================================================

def find_text(
    sentence_element: ET.Element,
) -> str | None:

    text_element = (
        sentence_element.find("text")
    )

    if (
        text_element is not None
        and text_element.text
    ):

        return text_element.text.strip()

    return None


def find_aspect_elements(
    sentence_element: ET.Element,
) -> list[ET.Element]:

    """
    Supports the common SemEval/MAMS style:

    <aspectTerms>
        <aspectTerm
            term="food"
            polarity="positive"
            from="4"
            to="8"
        />
    </aspectTerms>
    """

    aspects = []

    # Standard aspectTerm representation.
    aspects.extend(
        sentence_element.findall(
            ".//aspectTerm"
        )
    )

    # Defensive support for opinion-style XML.
    if not aspects:

        aspects.extend(
            sentence_element.findall(
                ".//Opinion"
            )
        )

    return aspects


def parse_aspect(
    element: ET.Element,
) -> dict | None:

    # --------------------------------------------------------
    # Aspect text
    # --------------------------------------------------------

    aspect = (
        element.attrib.get("term")
        or element.attrib.get("target")
        or element.attrib.get("aspect")
    )

    if aspect is None:
        return None

    aspect = aspect.strip()

    if not aspect:
        return None

    # --------------------------------------------------------
    # Polarity
    # --------------------------------------------------------

    polarity = (
        element.attrib.get("polarity")
        or element.attrib.get("sentiment")
    )

    polarity = normalize_polarity(
        polarity
    )

    # --------------------------------------------------------
    # Character positions
    # --------------------------------------------------------

    start_raw = (
        element.attrib.get("from")
        or element.attrib.get("start")
    )

    end_raw = (
        element.attrib.get("to")
        or element.attrib.get("end")
    )

    try:
        start = (
            int(start_raw)
            if start_raw is not None
            else None
        )
    except ValueError:
        start = None

    try:
        end = (
            int(end_raw)
            if end_raw is not None
            else None
        )
    except ValueError:
        end = None

    return {
        "aspect": aspect,
        "polarity": polarity,
        "from": start,
        "to": end,
    }


# ============================================================
# READ ONE XML SPLIT
# ============================================================

def read_xml_split(
    split_name: str,
    path: Path,
) -> list[dict]:

    print(
        f"\nReading {split_name}:"
    )

    print(path)

    tree = ET.parse(path)

    root = tree.getroot()

    sentences = []

    # MAMS/SemEval XML normally uses:
    #
    # <sentence id="...">
    #
    sentence_elements = (
        root.findall(".//sentence")
    )

    # Defensive support for uppercase XML tags.
    if not sentence_elements:

        sentence_elements = (
            root.findall(".//Sentence")
        )

    for sentence_element in (
        sentence_elements
    ):

        text = find_text(
            sentence_element
        )

        if not text:
            continue

        annotations = []

        for aspect_element in (
            find_aspect_elements(
                sentence_element
            )
        ):

            parsed = parse_aspect(
                aspect_element
            )

            if parsed is not None:
                annotations.append(
                    parsed
                )

        sentences.append(
            {
                "id":
                    sentence_element.attrib.get(
                        "id"
                    ),

                "split":
                    split_name,

                "text":
                    text,

                "aspects":
                    annotations,
            }
        )

    print(
        f"Sentences read: "
        f"{len(sentences)}"
    )

    return sentences


# ============================================================
# VALIDATE CHARACTER SPANS
# ============================================================

def validate_spans(
    sentences: list[dict],
) -> dict:

    checked = 0
    correct = 0
    incorrect = []

    for sentence in sentences:

        text = sentence["text"]

        for aspect in sentence[
            "aspects"
        ]:

            start = aspect["from"]
            end = aspect["to"]

            if (
                start is None
                or end is None
            ):
                continue

            checked += 1

            extracted = text[
                start:end
            ]

            if (
                normalize_text(extracted)
                ==
                normalize_text(
                    aspect["aspect"]
                )
            ):

                correct += 1

            else:

                incorrect.append(
                    {
                        "text": text,
                        "aspect":
                            aspect["aspect"],
                        "from": start,
                        "to": end,
                        "text_at_span":
                            extracted,
                    }
                )

    return {
        "checked": checked,
        "correct": correct,
        "incorrect":
            len(incorrect),
        "examples":
            incorrect[:20],
    }


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(
    all_sentences: list[dict],
    frozen_texts: set[str],
) -> dict:

    polarity_counts = Counter()

    split_sentence_counts = Counter()
    split_aspect_counts = Counter()

    mixed_sentences = []

    overlap = []

    unique_texts = set()

    duplicate_sentences = []

    total_aspects = 0

    sentences_without_aspects = 0

    for sentence in all_sentences:

        text = sentence["text"]

        normalized = normalize_text(
            text
        )

        split = sentence["split"]

        split_sentence_counts[
            split
        ] += 1

        if normalized in unique_texts:

            duplicate_sentences.append(
                {
                    "text": text,
                    "split": split,
                }
            )

        unique_texts.add(
            normalized
        )

        if normalized in frozen_texts:

            overlap.append(
                {
                    "text": text,
                    "split": split,
                }
            )

        aspects = sentence[
            "aspects"
        ]

        if not aspects:

            sentences_without_aspects += 1

        total_aspects += len(
            aspects
        )

        split_aspect_counts[
            split
        ] += len(
            aspects
        )

        sentiments = set()

        for aspect in aspects:

            polarity = aspect[
                "polarity"
            ]

            if polarity is not None:

                polarity_counts[
                    polarity
                ] += 1

                sentiments.add(
                    polarity
                )

        # ----------------------------------------------------
        # Important for Basha:
        #
        # Positive AND Negative aspect inside same sentence.
        # ----------------------------------------------------

        if (
            "positive" in sentiments
            and "negative" in sentiments
        ):

            mixed_sentences.append(
                sentence
            )

    span_validation = validate_spans(
        all_sentences
    )

    return {
        "total_sentences":
            len(all_sentences),

        "unique_sentences":
            len(unique_texts),

        "total_aspects":
            total_aspects,

        "sentences_without_aspects":
            sentences_without_aspects,

        "sentence_counts_by_split":
            dict(
                split_sentence_counts
            ),

        "aspect_counts_by_split":
            dict(
                split_aspect_counts
            ),

        "polarity_counts":
            dict(
                polarity_counts
            ),

        "mixed_sentences":
            len(mixed_sentences),

        "mixed_percentage":
            (
                len(mixed_sentences)
                / len(all_sentences)
                * 100
            )
            if all_sentences
            else 0.0,

        "frozen_benchmark_overlap":
            len(overlap),

        "frozen_overlap_examples":
            overlap[:20],

        "duplicate_sentences":
            len(
                duplicate_sentences
            ),

        "duplicate_examples":
            duplicate_sentences[:20],

        "span_validation":
            span_validation,

        "mixed_examples":
            mixed_sentences[:20],
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_analysis(
    analysis: dict,
) -> None:

    print("\n" + "=" * 70)

    print(
        "MAMS-ATSA DATASET ANALYSIS"
    )

    print("=" * 70)

    print(
        "\nTotal sentences:      ",
        analysis[
            "total_sentences"
        ],
    )

    print(
        "Unique sentences:     ",
        analysis[
            "unique_sentences"
        ],
    )

    print(
        "Total aspect terms:   ",
        analysis[
            "total_aspects"
        ],
    )

    print(
        "No-aspect sentences:  ",
        analysis[
            "sentences_without_aspects"
        ],
    )

    # --------------------------------------------------------
    # Splits
    # --------------------------------------------------------

    print(
        "\nSENTENCES BY SPLIT"
    )

    print("-" * 45)

    for split, count in (
        analysis[
            "sentence_counts_by_split"
        ].items()
    ):

        print(
            f"{split:<15} {count}"
        )

    print(
        "\nASPECTS BY SPLIT"
    )

    print("-" * 45)

    for split, count in (
        analysis[
            "aspect_counts_by_split"
        ].items()
    ):

        print(
            f"{split:<15} {count}"
        )

    # --------------------------------------------------------
    # Polarities
    # --------------------------------------------------------

    print(
        "\nPOLARITY COUNTS"
    )

    print("-" * 45)

    for polarity, count in (
        analysis[
            "polarity_counts"
        ].items()
    ):

        print(
            f"{polarity:<15} {count}"
        )

    # --------------------------------------------------------
    # Mixed
    # --------------------------------------------------------

    print(
        "\nMIXED SENTENCES"
    )

    print("-" * 45)

    print(
        "Positive + Negative:",
        analysis[
            "mixed_sentences"
        ],
    )

    print(
        "Percentage:         ",
        f"{analysis['mixed_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    print(
        "\nFROZEN BASHA BENCHMARK OVERLAP"
    )

    print("-" * 45)

    overlap = analysis[
        "frozen_benchmark_overlap"
    ]

    print(
        "Overlap sentences:",
        overlap,
    )

    if overlap == 0:

        print(
            "✓ No benchmark leakage."
        )

    else:

        print(
            "⚠ WARNING: Overlap exists."
        )

        print(
            "These sentences MUST be "
            "removed before training."
        )

    # --------------------------------------------------------
    # Spans
    # --------------------------------------------------------

    span = analysis[
        "span_validation"
    ]

    print(
        "\nCHARACTER SPAN VALIDATION"
    )

    print("-" * 45)

    print(
        "Spans checked:",
        span["checked"],
    )

    print(
        "Correct spans:",
        span["correct"],
    )

    print(
        "Incorrect spans:",
        span["incorrect"],
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    print(
        "\nDUPLICATE SENTENCES"
    )

    print("-" * 45)

    print(
        analysis[
            "duplicate_sentences"
        ]
    )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    print(
        "\nSAMPLE MIXED SENTENCES"
    )

    print("-" * 70)

    for index, sentence in enumerate(
        analysis[
            "mixed_examples"
        ][:5],
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{sentence['text']}"
        )

        for aspect in sentence[
            "aspects"
        ]:

            print(
                f"   "
                f"{aspect['aspect']} "
                f"→ "
                f"{aspect['polarity']}"
            )


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    analysis: dict,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analysis,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nDetailed inspection saved:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "BASHA — MAMS-ATSA XML INSPECTION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # 2. Frozen benchmark
    # --------------------------------------------------------

    frozen_texts = (
        load_frozen_texts()
    )

    print(
        f"\nFrozen Basha benchmark: "
        f"{len(frozen_texts)} sentences"
    )

    # --------------------------------------------------------
    # 3. Read XML
    # --------------------------------------------------------

    all_sentences = []

    for split_name, path in (
        MAMS_FILES.items()
    ):

        sentences = read_xml_split(
            split_name,
            path,
        )

        all_sentences.extend(
            sentences
        )

    # --------------------------------------------------------
    # 4. Analyze
    # --------------------------------------------------------

    analysis = analyze_dataset(
        all_sentences,
        frozen_texts,
    )

    # --------------------------------------------------------
    # 5. Print
    # --------------------------------------------------------

    print_analysis(
        analysis
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    save_result(
        analysis
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MAMS INSPECTION COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()