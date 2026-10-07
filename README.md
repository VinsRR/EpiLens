# EpiLens

EpiLens is a local-first Python toolkit for exploring scientific papers. It
parses and indexes documents, retrieves passages relevant to a question, and
can use a language model to produce evidence-linked answers or structured
classifications and extractions. It was built for epidemiology and public
health, but its workflows can be adapted to other research fields.

## Install

Requires Python 3.10 or newer:

```bash
python -m pip install epilens
```

## Try it on a paper

Replace `paper.pdf` with a PDF on your machine:

```bash
epilens inspect paper.pdf
epilens explore "What data sources were used?" --path paper.pdf --quality fast
```

These commands need no API key or database server. The first retrieval downloads and
caches an embedding model; `inspect` works fully offline.

To generate an answer, add a model provider. For example:

```bash
python -m pip install "epilens[gemini]"
epilens quickstart
epilens ask "What data sources were used?" --path paper.pdf --quality fast
```

`quickstart` guides provider setup. Other hosted providers and local Ollama
are supported; see [installation](https://github.com/VinsRR/EpiLens/wiki/Installation)
and [configuration](https://github.com/VinsRR/EpiLens/wiki/Configuration).

## Work with a collection

A workspace keeps papers, a local index, and task definitions together. Replace
`./papers` with a folder of your PDFs:

```bash
epilens init my-review
epilens index ./papers --workspace my-review
epilens explore "Which papers mention data repositories?" --workspace my-review
```

EpiLens also includes reusable classification and extraction tasks. Their
structured outputs and source passages are designed for review; validate a task
on examples from your corpus before using its results at scale.

## Learn more

- [Guided notebook tutorial](https://github.com/VinsRR/EpiLens/blob/main/notebooks/README.md) — a small, offline corpus
  exercise; the other notebooks are optional deep dives.
- [CLI quickstart](https://github.com/VinsRR/EpiLens/wiki/Local-CLI-Quickstart)
  and [command reference](https://github.com/VinsRR/EpiLens/wiki/CLI-Reference)
- [Workspaces](https://github.com/VinsRR/EpiLens/wiki/Workspaces) and
  [custom classification/extraction tasks](https://github.com/VinsRR/EpiLens/wiki/Declarative-Tasks)
- [Python API](https://github.com/VinsRR/EpiLens/wiki/Python-Usage),
  [Studio](https://github.com/VinsRR/EpiLens/wiki/Streamlit-UI), and
  [server deployment](https://github.com/VinsRR/EpiLens/wiki/Server-Mode-and-Docker)
- [Configuration](https://github.com/VinsRR/EpiLens/wiki/Configuration),
  [troubleshooting](https://github.com/VinsRR/EpiLens/wiki/Troubleshooting), and
  the [full wiki](https://github.com/VinsRR/EpiLens/wiki)

## Development

For a local development install, run `python -m pip install -e ".[dev]"` and
`pytest -q`. See the repository's [tests](https://github.com/VinsRR/EpiLens/blob/main/tests/README.md) for test suites.

## Citation and license

Citation metadata is in [CITATION.cff](https://github.com/VinsRR/EpiLens/blob/main/CITATION.cff).
EpiLens is licensed under [AGPL-3.0-only](https://github.com/VinsRR/EpiLens/blob/main/LICENSE.txt).
