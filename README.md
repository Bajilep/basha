# BASHA

**An English Sentiment Analysis Library**

BASHA is a Python library for English sentiment analysis. It aims to provide deeper sentiment understanding through sentiment classification, aspect-based analysis, sarcasm detection, discourse analysis, semantic conflict detection, and long-document analysis.

## Features

- **Sentiment Analysis:** Analyze positive, negative, neutral, and mixed sentiment.
- **Aspect-Based Sentiment Analysis:** Identify sentiment associated with individual aspects or topics.
- **Sarcasm Detection:** Analyze potentially sarcastic expressions.
- **Mixed Sentiment Detection:** Identify contrasting opinions within text.
- **Discourse Analysis:** Examine relationships between clauses and sentences.
- **Semantic Conflict Detection:** Identify conflicting sentiment or meaning.
- **Long-Text Analysis:** Analyze longer reviews and documents by processing smaller text segments.

*Feature availability and model requirements may vary by module.*

## Installation

Install BASHA using pip:

```bash
pip install basha
```

## Quick Start

```python
from basha.fusion.english_fusion import EnglishFusion

analyzer = EnglishFusion()

text = "The food was excellent, but the service was terrible."

result = analyzer.analyze(text)

print(result)
```

## Project Structure

```text
basha/
├── src/
│   └── basha/
│       ├── aspects/
│       ├── fusion/
│       ├── longtext/
│       ├── sarcasm/
│       ├── semantics/
│       └── sentiment/
├── tests/
├── examples/
├── README.md
├── LICENSE
└── pyproject.toml
```

## Requirements

- Python 3.10 or later
- Dependencies listed in `pyproject.toml`

Some features may require additional pretrained model resources.

## Running Tests

Clone the repository and install development dependencies:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd basha
pip install -e ".[dev]"
pytest
```

## Author

**BAJIL MOHAMMED .EP**

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.