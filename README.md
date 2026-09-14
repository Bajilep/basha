# Basha

Basha is a lightweight, explainable and customizable Python library for sentiment and emotion analysis.

It analyses English text and returns:

- Sentiment label: positive, negative or neutral
- Continuous compound score from `-1` to `+1`
- Positive, negative and neutral proportions
- Emotion information
- Sarcasm information
- Per-word explanations
- Social-media normalization information

## Features

Basha currently supports:

- Weighted sentiment vocabulary
- Positive, negative and neutral classification
- Emotion detection
- Negation handling
- Intensifiers and diminishers
- Capital-letter emphasis
- Punctuation emphasis
- Contrast words such as `but`
- Emoji and emoticon sentiment
- Spelling correction for sentiment words
- Complete sentiment phrases
- Context-dependent words
- Basic sarcasm detection
- Social-media slang
- Hashtag conversion
- Username and URL removal
- Batch prediction
- Custom sentiment words
- Explainable output

## Installation

After the package is published, install it using:

```bash
pip install basha-sentiment
```

For local development, open the terminal in the project folder and run:

```bash
python -m pip install --editable .
```

## Basic usage

```python
from basha import bashaanalyser


analyzer = bashaanalyser()

result = analyzer.predict(
    "This product is excellent"
)

print(result)
```

## Get the sentiment label

```python
from basha import bashaanalyser


analyzer = bashaanalyser()

result = analyzer.predict(
    "This product is excellent"
)

print(result["label"])
```

Output:

```text
positive
```

## Standard class-name usage

Python classes normally use capital letters. Basha also supports:

```python
from basha import BashaAnalyser


analyzer = BashaAnalyser()

result = analyzer.predict(
    "This is amazing!"
)

print(result["label"])
```

## Example output

```python
{
    "text": "This product is very good!",
    "label": "positive",
    "emotion": {
        "label": "joy",
        "confidence": 0.75
    },
    "scores": {
        "compound": 0.72,
        "positive": 0.25,
        "negative": 0.0,
        "neutral": 0.75
    },
    "flags": {
        "sarcasm": False,
        "normalized_text": "This product is very good!",
        "social_normalizations": []
    },
    "explanation": [
        {
            "token": "good",
            "base_score": 2.0,
            "final_score": 3.0,
            "rules": [
                "intensifier: very"
            ]
        }
    ]
}
```

The exact numerical values can change when the vocabulary and rules are updated.

## Negation

```python
print(
    analyzer.predict(
        "This product is not good"
    )["label"]
)
```

Output:

```text
negative
```

Basha also handles positive negation:

```python
print(
    analyzer.predict(
        "This product is not bad"
    )["label"]
)
```

Output:

```text
positive
```

## Intensifiers and diminishers

```python
normal = analyzer.predict("good")
strong = analyzer.predict("very good")
weak = analyzer.predict("slightly good")

print(normal["scores"]["compound"])
print(strong["scores"]["compound"])
print(weak["scores"]["compound"])
```

`very good` should have a stronger positive score than `good`, while `slightly good` should have a weaker positive score.

## Contrast handling

```python
result = analyzer.predict(
    "The camera is good, but the battery is terrible"
)

print(result["label"])
```

Output:

```text
negative
```

The part appearing after `but` receives more importance.

## Emoji and emoticon support

```python
print(
    analyzer.predict(
        "I love this phone 😍"
    )["label"]
)

print(
    analyzer.predict(
        "This is bad :("
    )["label"]
)
```

Output:

```text
positive
negative
```

## Spelling correction

Basha can safely correct certain misspelled sentiment words:

```python
result = analyzer.predict(
    "This product is graet"
)

print(result["label"])
print(result["explanation"])
```

The word `graet` can be interpreted as `great`.

Basha does not automatically correct every unknown word. It corrects only sufficiently clear spelling mistakes to reduce incorrect corrections.

## Sarcasm detection

```python
result = analyzer.predict(
    "Excellent service, yeah right."
)

print(result["label"])
print(result["flags"]["sarcasm"])
print(result["flags"]["sarcasm_reasons"])
```

Possible output:

```text
negative
True
['sarcasm marker: yeah right']
```

Sarcasm detection is rule-based and cannot understand every form of sarcasm.

## Social-media text

```python
result = analyzer.predict(
    "@company this product is gr8! #AmazingProduct"
)

print(result["label"])
print(result["flags"]["normalized_text"])
print(result["flags"]["social_normalizations"])
```

Basha can:

- Convert `gr8` to `great`
- Convert `luv` to `love`
- Split `#AmazingProduct` into `amazing product`
- Ignore usernames such as `@company`
- Ignore URLs
- Normalize repeated letters

## Emotion detection

```python
result = analyzer.predict(
    "I am extremely happy today"
)

print(result["emotion"])
```

Basha can identify emotions supported by its emotion vocabulary, such as:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust
- Neutral or uncertain emotion

## Context-dependent words

Some words have different meanings in different situations:

```python
print(
    analyzer.predict(
        "The movie was sick"
    )["label"]
)

print(
    analyzer.predict(
        "I feel sick"
    )["label"]
)
```

Possible output:

```text
positive
negative
```

Other context-dependent words include words such as `sharp`, `hot`, `sweet`, `fire` and `killer`.

## Custom words

You can add a word to one analyzer:

```python
from basha import bashaanalyser


analyzer = bashaanalyser()

analyzer.add_word(
    "supercalifragilistic",
    3.5
)

result = analyzer.predict(
    "This is supercalifragilistic"
)

print(result["label"])
```

The custom score must be between `-4.0` and `+4.0`.

## Batch prediction

```python
from basha import bashaanalyser


analyzer = bashaanalyser()

results = analyzer.predict_batch([
    "This product is excellent",
    "This product is terrible",
    "The parcel arrived today"
])

for result in results:
    print(result["label"])
```

Output:

```text
positive
negative
neutral
```

## Running tests

Install development tools:

```bash
python -m pip install --editable ".[dev]"
```

Run all tests:

```bash
python -m pytest
```

Passing tests prove that the tested behaviours work correctly. They do not prove complete real-world accuracy.

## Benchmarking

Optional benchmark dependencies can be installed using:

```bash
python -m pip install --editable ".[benchmark]"
```

A benchmark can compare Basha with VADER and TextBlob using:

- Total execution time
- Average time per sentence
- Sentences processed per second

Do not claim that Basha is faster until benchmark results confirm it.

## Project structure

```text
basha/
├── src/
│   └── basha/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── context.py
│       ├── emotions.py
│       ├── lexicon.py
│       ├── modifiers.py
│       ├── phrases.py
│       ├── sarcasm.py
│       ├── social.py
│       ├── spelling.py
│       ├── symbols.py
│       └── tokenizer.py
├── tests/
│   └── test_sentiment.py
├── example.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Limitations

Basha is currently a rule-based English sentiment-analysis library.

Known limitations include:

- It does not learn automatically from new data
- Sarcasm detection is limited to recognized patterns
- Context understanding is limited to defined rules
- Spelling correction does not correct every unknown word
- Emotion detection depends on its emotion vocabulary
- Mixed sentiment may be simplified into one overall label
- Multilingual text is not yet fully supported
- Real-world accuracy must still be measured using labelled datasets

## Development status

Basha is currently under active development. Its vocabulary, rules, output format and API may change before version `1.0.0`.

## Author

Bajil Mohammed E.P.

## License

This project is licensed under the MIT License.