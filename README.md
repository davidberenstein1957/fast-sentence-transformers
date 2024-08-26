# Fast Sentence Transformers
This repository contains code to run faster feature extractors using tools like quantization, optimization and `ONNX`. Just run your model much faster, while using less of memory. There is not much to it!

[![Python package](https://github.com/Pandora-Intelligence/fast-sentence-transformers/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Pandora-Intelligence/fast-sentence-transformers/actions/workflows/python-package.yml)
[![Current Release Version](https://img.shields.io/github/release/pandora-intelligence/fast-sentence-transformers.svg?style=flat-square&logo=github)](https://github.com/pandora-intelligence/fast-sentence-transformers/releases)
[![pypi Version](https://img.shields.io/pypi/v/fast-sentence-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/fast-sentence-transformers/)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/fast-sentence-transformers?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/fast-sentence-transformers/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Install

```bash
pip install fast-sentence-transformers
```

## Quickstart

```python

from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

# use any sentence-transformer
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", quantize=True)

encoder.encode("Hello hello, hey, hello hello")
encoder.encode(["Life is too short to eat bad food!"] * 2)
```

## Benchmark

Non-exact, indicative benchmark for CPU usage with smallest and largest model on `sentence-transformers`

| model                                 | Type   | default | ONNX | ONNX+quantized | ONNX+GPU |
| ------------------------------------- | ------ | ------- | ---- | -------------- | -------- |
| paraphrase-albert-small-v2            | memory | 1x      | 1x   | 1x             | 1x       |
|                                       | speed  | 1x      | 2x   | 5x             | 20x      |
| paraphrase-multilingual-mpnet-base-v2 | memory | 1x      | 1x   | 4x             | 4x       |
|                                       | speed  | 1x      | 2x   | 5x             | 20x      |

## Shout-Out

This package heavily leans on https://www.philschmid.de/optimize-sentence-transformers.
