from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "absa_final_train.json"
VALIDATION_FILE = DATA_DIR / "absa_final_validation.json"


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def find_occurrences(
    text: str,
    term: str,
):

    text_lower = text.lower()
    term_lower = term.lower()

    results = []

    start = 0

    while True:

        index = text_lower.find(
            term_lower,
            start,
        )

        if index == -1:
            break

        results.append(
            (
                index,
                index + len(term),
            )
        )

        start = index + 1

    return results


def inspect_split(
    examples,
    split_name,
):

    print("\n" + "=" * 80)
    print(split_name)
    print("=" * 80)

    missing = []
    overlaps = []

    for example_index, example in enumerate(
        examples
    ):

        text = example["text"]

        resolved = []

        used = set()

        # ----------------------------------------------------
        # Resolve all aspect spans
        # ----------------------------------------------------

        for aspect_index, aspect in enumerate(
            example.get("aspects", [])
        ):

            term = str(
                aspect.get(
                    "aspect",
                    ""
                )
            ).strip()

            polarity = aspect.get(
                "polarity"
            )

            start = aspect.get("from")
            end = aspect.get("to")

            span = None

            # Existing valid span
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
            ):

                actual = text[start:end]

                if (
                    actual.strip().lower()
                    ==
                    term.lower()
                ):
                    span = (
                        start,
                        end,
                    )

            # Recover missing span
            if span is None:

                occurrences = (
                    find_occurrences(
                        text,
                        term,
                    )
                )

                for candidate in occurrences:

                    if candidate not in used:

                        span = candidate
                        break

            if span is None:

                missing.append(
                    {
                        "example_index":
                            example_index,

                        "text":
                            text,

                        "aspect":
                            term,

                        "polarity":
                            polarity,

                        "source":
                            example.get(
                                "source"
                            ),

                        "all_aspects":
                            example.get(
                                "aspects",
                                [],
                            ),
                    }
                )

                continue

            used.add(span)

            resolved.append(
                {
                    "aspect_index":
                        aspect_index,

                    "term":
                        term,

                    "polarity":
                        polarity,

                    "start":
                        span[0],

                    "end":
                        span[1],
                }
            )

        # ----------------------------------------------------
        # Detect overlapping aspect spans
        # ----------------------------------------------------

        for i in range(
            len(resolved)
        ):

            first = resolved[i]

            for j in range(
                i + 1,
                len(resolved)
            ):

                second = resolved[j]

                overlap = (
                    first["start"]
                    < second["end"]
                    and
                    second["start"]
                    < first["end"]
                )

                if not overlap:
                    continue

                overlaps.append(
                    {
                        "example_index":
                            example_index,

                        "text":
                            text,

                        "source":
                            example.get(
                                "source"
                            ),

                        "first":
                            first,

                        "second":
                            second,

                        "all_aspects":
                            example.get(
                                "aspects",
                                [],
                            ),
                    }
                )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    print(
        f"\nSPAN NOT FOUND: "
        f"{len(missing)}"
    )

    print("-" * 80)

    for number, item in enumerate(
        missing,
        start=1,
    ):

        print(
            f"\nMISSING #{number}"
        )

        print(
            "Source:",
            item["source"],
        )

        print(
            "Text:"
        )

        print(
            item["text"]
        )

        print(
            "\nMissing aspect:"
        )

        print(
            f"  {item['aspect']} "
            f"→ {item['polarity']}"
        )

        print(
            "\nAll annotations:"
        )

        for aspect in item[
            "all_aspects"
        ]:

            print(
                " ",
                aspect,
            )

    # --------------------------------------------------------
    # Overlaps
    # --------------------------------------------------------

    print(
        f"\n\nOVERLAPPING ASPECT SPANS: "
        f"{len(overlaps)}"
    )

    print("-" * 80)

    for number, item in enumerate(
        overlaps,
        start=1,
    ):

        print(
            f"\nOVERLAP #{number}"
        )

        print(
            "Source:",
            item["source"],
        )

        print(
            "Text:"
        )

        print(
            item["text"]
        )

        first = item["first"]
        second = item["second"]

        print(
            "\nAspect A:"
        )

        print(
            f"  {first['term']} "
            f"→ {first['polarity']} "
            f"[{first['start']}:"
            f"{first['end']}]"
        )

        print(
            "Aspect B:"
        )

        print(
            f"  {second['term']} "
            f"→ {second['polarity']} "
            f"[{second['start']}:"
            f"{second['end']}]"
        )

        print(
            "\nText fragments:"
        )

        print(
            " A =",
            item["text"][
                first["start"]:
                first["end"]
            ],
        )

        print(
            " B =",
            item["text"][
                second["start"]:
                second["end"]
            ],
        )

    return {
        "missing":
            len(missing),

        "overlapping_pairs":
            len(overlaps),
    }


def main():

    print("=" * 80)
    print(
        "BASHA — GAUNEG ALIGNMENT "
        "ERROR INSPECTION"
    )
    print("=" * 80)

    train = load_json(
        TRAIN_FILE
    )

    validation = load_json(
        VALIDATION_FILE
    )

    train_result = inspect_split(
        train,
        "TRAINING DATA",
    )

    validation_result = inspect_split(
        validation,
        "VALIDATION DATA",
    )

    print(
        "\n\n" + "=" * 80
    )

    print("SUMMARY")

    print("=" * 80)

    print(
        "\nTraining:"
    )

    print(
        "  Missing spans:",
        train_result["missing"],
    )

    print(
        "  Overlapping pairs:",
        train_result[
            "overlapping_pairs"
        ],
    )

    print(
        "\nValidation:"
    )

    print(
        "  Missing spans:",
        validation_result[
            "missing"
        ],
    )

    print(
        "  Overlapping pairs:",
        validation_result[
            "overlapping_pairs"
        ],
    )

    print(
        "\nDo NOT modify the dataset yet."
    )


if __name__ == "__main__":
    main()