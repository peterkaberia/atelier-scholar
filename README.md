# 🏛️ Atelier

> An open-source, AI-powered academic research synthesizer and literature copilot — a Consensus.app-style alternative you can run yourself.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/damurka/atelier/blob/main/atelier-demo.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Dash](https://img.shields.io/badge/UI-Plotly_Dash-informational.svg)]()

**Atelier** takes a natural-language research question, plans engine-specific boolean queries with an LLM, concurrently searches six academic databases, ranks candidates with a local sparse-vector model weighted by citation count and study design, extracts structured per-paper metadata, and writes a fully cited, abstract-style synthesis — all through a reactive Dash single-page app with a resumable, checkpointed pipeline underneath.

## ✨ Core Features

* 🧠 **LLM-Orchestrated Pipeline:** A [LangGraph](https://github.com/langchain-ai/langgraph) state machine plans queries, searches, ranks, extracts, and synthesizes - checkpointed via SQLite so an interrupted run resumes exactly where it left off instead of restarting from scratch.
* 🔍 **Six-Engine Retrieval:** Searches **PubMed**, **Europe PMC**, **OpenAlex**, **Semantic Scholar**, **Crossref**, and **arXiv** in parallel, merges and deduplicates by a global fingerprint (DOI/PMID/PMCID/arXiv ID), and surfaces per-engine failures instead of silently under-reporting.
* 📊 **Quality-Weighted Ranking:** Relevance scoring blends SPLADE sparse-vector similarity with citation count, recency, and a study-type evidence tier (systematic review > RCT > cohort > case report, etc.) - not just topical match.
* 🧭 **Atelier Meter:** For yes/no-shaped research questions, a weighted agreement bar shows how the literature actually leans, with each paper's vote weighted by its evidence tier and citation count rather than counted equally.
* ✍️ **Abstract-Style Synthesis:** Background → Evidence Synthesis → Key Findings → Conclusion, written in real academic-abstract prose that explicitly characterizes how strong and consistent the evidence is - not just a list of findings.
* 💬 **Context-Aware Follow-Ups:** Ask questions, request a rewrite of your own pasted draft with citations woven in, or ask for a specific deliverable ("write a 300-word abstract", "write an introduction") - each handled with its own genre-appropriate prompt. Follow-ups first check papers your original search already found but didn't extract before falling back to a full new external search.
* 📄 **Per-Paper AI Summaries:** Click any paper for a topic-focused summary, its full author list, a link to the journal (via DOI), and an in-app PDF viewer - summaries are cached per session so re-opening a paper is instant.
* 📥 **Reference Export:** Download citations as BibTeX or RIS, either the whole turn's reference list (selection-aware) or a single paper at a time - compatible with Zotero, EndNote, and Mendeley.
* 🔄 **Load More:** Pull additional already-ranked candidates into an existing turn's evidence library on demand, without re-running the search.
* 🎨 **Reactive Dash UI:** TailwindCSS-styled frontend with interactive citation hover-popups, study-type filter chips, live session status, and asynchronous background execution that survives page navigation.
* 🔐 **Encrypted, UI-Configurable Keys:** Add LLM and academic-API keys from the in-app Settings page - encrypted at rest, no restart required.

## 🛠️ Tech Stack

* **Pipeline:** [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with a `SqliteSaver` checkpointer, keyed per session *and* per follow-up topic so genuinely different searches never replay stale results.
* **Frontend:** Plotly Dash, TailwindCSS, custom clientside JavaScript (citation popovers, PDF viewer, filter chips)
* **Backend:** Python, SQLAlchemy ORM
* **Database:** SQLite (WAL mode) - structured metadata, SPLADE sparse vectors, and full-text chunks for RAG-grounded extraction/synthesis
* **Embeddings:** SPLADE sparse encoder (`sentence-transformers`) for relevance scoring and chunk-level retrieval
* **LLM Providers:** Any OpenAI-compatible provider - OpenRouter, LM Studio (local), or others - configured per-request as `provider:model`
* **APIs:** NCBI E-utilities, Europe PMC REST, OpenAlex, Semantic Scholar Graph, Crossref, arXiv

## 🚀 Installation

Atelier is a [uv](https://docs.astral.sh/uv/) project - dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

### 1. Clone the repository

```bash
git clone https://github.com/damurka/atelier.git
cd atelier
```

### 2. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
uv sync
```

This creates `.venv` and installs every pinned dependency into it (including `diskcache` for Dash's background callbacks, and `langgraph-checkpoint-sqlite` for pipeline resumability) - no separate venv-creation or activation step needed; `uv run` (below) uses `.venv` automatically.

## ⚙️ Configuration

Atelier requires at least one LLM provider key to run; academic-API keys (OpenAlex, Semantic Scholar) are optional and only raise rate limits. PubMed, Europe PMC, Crossref, and arXiv need no key at all.

The easiest way to configure keys is the in-app **Settings** page (`/settings`) once the server is running - keys are encrypted at rest (see `docs/adr/0001-local-encryption-key-for-ui-settings.md`), take priority over `.env`, and apply immediately with no restart. `.env` remains a supported fallback:

1. Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

2. Add your API keys to the `.env` file:

```.env
# Academic APIs (optional - raise rate limits, don't gate search)
OPENALEX_API_KEY=your_openalex_key
SEMANTIC_SCHOLAR_API_KEY=your_s2_key

# LLM Provider (at least one required - see llm/model_catalog.py for the
# full provider list; models are selected in-app as "provider:model")
OPENROUTER_API_KEY=your_openrouter_key
```

## 💻 Usage

Start the Atelier server:

```bash
uv run python app.py
```

1. Open your browser and navigate to http://localhost:8050.
2. Type a research question into the search bar (e.g., ***"What are the socioeconomic barriers to breast cancer screening in sub-Saharan Africa?"***).
3. Atelier plans the search, fetches papers from all six engines, ranks and extracts the most relevant ones, and generates a fully cited, abstract-style synthesis - plus an Atelier Meter if the question is yes/no-shaped.
4. Ask follow-ups in the same thread: pointed questions, "rewrite this paragraph with the evidence," or "write me a 300-word abstract."

## 🏗️ Architecture Overview

* `pipeline/`: The research workflow as a LangGraph `StateGraph` (`graph.py`/`nodes.py`), orchestrated by `orchestrator.py`. Checkpointed via SQLite so an interrupted or resumed run picks up from its last completed node.
* `search/`: Object-oriented API wrappers for all six engines (`paper.py`), a throttled/retrying `HttpClient`, and the SPLADE-based sparse-vector scorer (`sparse_encoder.py`).
* `llm/`: The LLM engine - query planning, batched per-paper extraction, abstract-style synthesis, the Atelier Meter, conversational follow-ups, and per-paper summaries.
* `database/`: SQLAlchemy models and a repository layer abstracting all persistence, including hybrid text/vector local recall and full chat history.
* `ui/`: Dash layouts, Tailwind styling, and reactive callbacks - background jobs are gated through a session-aware store so a job for one session can never silently overwrite what a user has since navigated to in the same tab.
* `docs/adr/`: Architecture decision records for the non-obvious design choices.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See LICENSE for more information.
