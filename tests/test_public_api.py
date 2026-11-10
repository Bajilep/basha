
import basha


TEST_SENTENCES = [
    "The food was excellent and the staff were friendly.",
    "The service was terrible and the food was cold.",
    "The product arrived yesterday.",
    "The camera quality is excellent, but the battery is disappointing.",
    "Wow, what an amazing service! I waited three hours.",
    "I love this product. It works perfectly.",
    "I am extremely disappointed with this purchase.",
]


def test_public_functions():

    for text in TEST_SENTENCES:

        print("\n" + "=" * 60)
        print("INPUT:", text)

        print("\nBASIC SENTIMENT:")
        print(basha.sentiment(text))

        print("\nFULL ANALYSIS:")
        print(basha.analyze(text))


def test_document_analysis():

    document = "\n".join(TEST_SENTENCES)

    print("\n" + "=" * 60)
    print("DOCUMENT ANALYSIS:")
    print(basha.analyze_document(document))


if __name__ == "__main__":
    test_public_functions()
    test_document_analysis()
def test_long_document_analysis():
    from basha.api.public import _get_document_analyzer

    long_document = "\n".join(TEST_SENTENCES * 80)

    result = basha.analyze_document(long_document)

    # Inspect internal chunk information for testing only.
    internal_result = _get_document_analyzer().analyze(long_document)

    assert internal_result["longtext"]["chunk_count"] > 1
    assert result["sentiment"]["label"] in {
        "positive", "negative", "neutral", "mixed"
    }
    assert len(result["sentences"]) > 100

    print("\nLONG DOCUMENT RESULT:")
    print("Sentiment:", result["sentiment"])
    print("Number of sentences:", len(result["sentences"]))
    print("Number of chunks:", internal_result["longtext"]["chunk_count"])