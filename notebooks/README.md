# EpiLens notebooks

Start with **[01: guided tutorial](01_core_quickstart.ipynb)**. It is a short,
complete exercise: inspect two synthetic papers, build a local index, retrieve
candidate passages, apply one classification task to both papers, review the
evidence, and count the labels. At the review step, ask participants whether
each label is supported by the displayed passage. The other notebooks are
optional explorations, not prerequisites for 01.

The examples run offline. `epilens_nb.py` supplies a tiny keyword embedder and
scripted JSON responses, so no model, API key, service, or network download is
needed. **The scripted labels and extracted items are supplied by the notebooks;
they are not predictions.** Keyword scores illustrate the retrieval interface,
not retrieval quality. A valid JSON schema checks output shape. The notebooks
show candidate passages for review; model provenance, when present, gives
another trace to inspect. Neither establishes that a classification or
extraction is correct.

## Run the notebooks

From a clone of this repository:

```bash
python -m pip install -e .
python -m pip install jupyterlab
jupyter lab notebooks/
```

Run each notebook top to bottom. The setup cells work when the kernel starts in
the repository root or in `notebooks/`. Scratch indexes and stores are created
in temporary directories. The included PDFs are small open examples, separate
from the manuscript's research corpus.

## Choose a follow-up

| Notebook | What it teaches |
| --- | --- |
| [01 — Guided tutorial](01_core_quickstart.ipynb) | One fixed classification scheme across a tiny corpus, with an evidence check and a count. Start here. |
| [02 — Indexing and retrieval](02_indexing_and_retrieval.ipynb) | Chunking, stored payloads, paper-scoped retrieval, and section filters. |
| [03 — PDF ingestion](03_pdf_ingestion.ipynb) | Parse sample PDFs and inspect extraction quality before indexing. |
| [04 — PDF retrieval](04_pdf_retrieval_and_workflows.ipynb) | Search the parsed PDF corpus and inspect retrieved passages. |
| [05 — Precision Miner](05_precision_miner_workflow.ipynb) | Extract a per-paper list of candidate data sources and check its support. |
| [06 — Custom miner](06_data_source_extraction.ipynb) | Define another extraction task with a declarative `TaskSpec`. |
| [07 — Custom classifier](07_classification_workflow.ipynb) | Compare a built-in classifier with a small declarative task. |
| [08 — Service runtime](08_service_runtime.ipynb) | Call the same `explore`, `classify`, and `precision_mine` verbs used by the app. |
| [09 — Workspaces](09_workspaces.ipynb) | Put configuration and task specs in a portable workspace folder. |

For a live session, **01 alone is the teaching path**. Use 02–05 when the group
wants to examine a specific stage, and 06–09 for task authoring or application
integration. A classifier's labels may be aggregated only after validation on
labelled examples and review of disagreements. A miner's output is a per-paper
worklist to verify, not a distribution to count without review. Data access
labels, when used, describe what a paper *reports*; they do not check whether a
link or dataset is currently accessible.

## Moving to real data

For a model-backed run, replace the demo embedder and scripted generator with
`EmbedderFactory.get_embedder(...)` and `LLMGenerator`, and configure a provider.
The default `fastembed` path downloads a model on first use. Check parsing and
retrieved passages on a sample of your own PDFs before processing a corpus,
then validate the task against manually labelled papers. The
[main README](../README.md) covers the CLI, API, UI, and workspace setup.
