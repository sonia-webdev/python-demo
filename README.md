# Named Entity Recognition with Word Embeddings

## Project problem

This project identifies and classifies **people (PER), organisations (ORG),
locations (LOC), and dates (DATE)** in text. For example, in
`Mary joined Acme Corporation in 2021.`, it should recognize `Mary` as a person,
`Acme Corporation` as an organisation, and `2021` as a date.

This is a first-project starter, not a continuation of a previous project. It
includes a small, hand-labelled CoNLL-format dataset so the pipeline can run
without downloading a dataset or a pre-trained model. The sample data is for
learning and demonstrating the workflow; it is too small to support claims
about production-quality NER.

## Embedding approach

The new model trains **32-dimensional corpus-specific embeddings** from the
training split. It counts nearby-word co-occurrences, applies positive pointwise
mutual information (PPMI), then uses truncated singular-value decomposition
(SVD) to produce dense word vectors. Token features combine the current word's
vector, the average vectors of its immediate neighbours, and basic word-shape
signals. This approach needs no large model download and makes it possible to
inspect and explain how embeddings are learned. Since this starter corpus is
small, these embeddings are not a substitute for embeddings trained on a large
general-language corpus.

The comparison baseline is a sparse bag-of-features token classifier using the
current token, nearby words, prefixes/suffixes, casing, and token shape. Both
models use logistic regression and are evaluated on the same held-out sentences.

## Run the project

Requires Python 3.10+.

```powershell
py -m pip install -r requirements.txt
py -m ner_embeddings
```

The command prints a JSON report and saves it to `reports/results.json`. To use
another CoNLL-formatted split:

```powershell
py -m ner_embeddings --train path\to\train.conll --test path\to\test.conll
```

## Open the interactive demo in Chrome

Install the requirements if you have not already, then start the local web app
from the extracted project folder:

```powershell
py -m pip install -r requirements.txt
py -m ner_embeddings.web_server
```

Open `http://127.0.0.1:8765` in Chrome. Enter text to compare both models'
predictions, and see the held-out test metrics below. The server binds to your
own computer only; press `Ctrl+C` in PowerShell to stop it. Choose from eight
sample texts in the dropdown, or select **Analyze all 8 examples** to see both
models' extracted entities side by side for every sample.

## Deploy on Streamlit Community Cloud

The repository includes `streamlit_app.py` as its Streamlit entrypoint. Push
the project branch to GitHub, then sign in at
[share.streamlit.io](https://share.streamlit.io/), choose **Create app**,
select this repository and the `ner-entity-extraction` branch, and set
`streamlit_app.py` as the main file. Streamlit Cloud installs packages from
`requirements.txt` automatically. The app needs no API keys or secrets.

Each non-empty CoNLL row contains a token, a space, and a BIO tag (`O`,
`B-PER`, `I-PER`, etc.); blank lines separate sentences. The test set must be
separate from the training set.

## Results

On the bundled split (30 training sentences and 12 test sentences), the
generated report gives the following **exact entity-span** scores. The test
support is 7 PER, 6 ORG, 8 LOC, and 7 DATE spans.

| Entity | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| PER | Sparse baseline | 1.000 | 0.857 | 0.923 |
| PER | Word embeddings | 0.455 | 0.714 | 0.556 |
| ORG | Sparse baseline | 0.500 | 0.667 | 0.571 |
| ORG | Word embeddings | 0.333 | 0.500 | 0.400 |
| LOC | Sparse baseline | 0.875 | 0.875 | 0.875 |
| LOC | Word embeddings | 0.286 | 0.500 | 0.364 |
| DATE | Sparse baseline | 0.667 | 0.857 | 0.750 |
| DATE | Word embeddings | 0.667 | 0.286 | 0.400 |
| **Micro average** | **Sparse baseline** | **0.742** | **0.821** | **0.780** |
| **Micro average** | **Word embeddings** | **0.378** | **0.500** | **0.431** |

**Discussion:** on this starter dataset, the embedding version did not improve
over the sparse baseline. The sparse model can use token spelling, nearby
words, and capitalization directly; in contrast, the embeddings are learned
from only 30 short training sentences. Test-set names absent from training have
no learned vector, and independent token predictions can split multiword
entities incorrectly. For example, the baseline correctly tags `United
Kingdom` as `B-LOC I-LOC`, while the embedding model predicts `B-PER B-LOC`.
This comparison is useful for the assignment because it shows both the measured
change and a concrete failure case, rather than claiming an improvement that
the test results do not support.

The same exact-span metrics and held-out sentences are used for both models.
The full generated report includes per-type support and token-level gold and
predicted tags for every test example. The split is tiny and hand-labelled, so
these scores are illustrative only, not a reliable estimate of real-world
performance. A sensible next step is to replace or expand the starter data
with a larger, representative annotated dataset and document its source and
split method.

## Project structure

- `data/train.conll`, `data/test.conll` — small labelled starter dataset
- `ner_embeddings/embeddings.py` — PPMI/SVD embedding training
- `ner_embeddings/features.py` — sparse baseline and embedding feature builders
- `ner_embeddings/metrics.py` — BIO decoding and exact entity-span metrics
- `ner_embeddings/cli.py` — training, evaluation, and report generation
- `streamlit_app.py` — Streamlit interactive application and sample analysis
- `tests/test_ner.py` — focused tests for data loading, embeddings, and metrics

Run tests with:

```powershell
py -m unittest discover -s tests
```

## Discussion and next steps

The embedding model represents similarity through neighbouring-word usage,
whereas the sparse baseline can memorize useful names and local patterns. The
comparison report makes it possible to see whether the dense representation
helps each entity type; the embedding model is not expected to win on this tiny
corpus. Rare and unseen names, multi-token entities, and dates in unfamiliar
formats are likely failure cases. Next, replace or expand the sample with a
larger annotated dataset, keep a reproducible held-out split, and compare these
models on that same split. A pretrained FastText or transformer embedding can
be explored as a further extension.
