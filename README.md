# 🏛️ Atelier

> An open-source, AI-powered academic research synthesizer and literature copilot.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Dash](https://img.shields.io/badge/UI-Plotly_Dash-informational.svg)]()

**Atelier** is an enterprise-grade research orchestrator. It takes a natural language research question, uses an LLM to plan optimized boolean queries, concurrently fetches papers from major academic databases, scores them using local vector math, and extracts structured full-text metadata (Sample Size, Methods, Outcomes) into a sleek, reactive UI.

## ✨ Core Features

* 🧠 **LLM Orchestration:** Automatically plans engine-specific queries and extracts permanent, structured metadata directly from full-text XML.
* 🔍 **Multi-Engine Retrieval:** Concurrently fetches from **PubMed**, **Europe PMC**, **OpenAlex**, and **Semantic Scholar**.
* 🗄️ **Local Vector Knowledge Graph:** Uses a highly optimized SQLite database (via Peewee) to store SPLADE sparse vectors, enabling instant, zero-API-cost semantic recall of previously read literature.
* 🛡️ **Enterprise Networking:** Built-in thread-safe throttling, exponential backoff, and robust `requests` session management to prevent rate-limit crashes.
* 🎨 **Reactive Dash UI:** A beautiful, TailwindCSS-powered frontend featuring interactive citation hover-popups, asynchronous background execution, and a ChatGPT-style conversational feed.

## 🛠️ Tech Stack

* **Frontend:** Plotly Dash, TailwindCSS, custom Clientside Javascript
* **Backend:** Python, Peewee ORM
* **Database:** SQLite (with custom JSON field serialization for vectors)
* **APIs:** NCBI E-utilities, Europe PMC REST, OpenAlex, Semantic Scholar Graph

## 🚀 Installation

### 1. Clone the repository

```bash
git clone [https://github.com/](https://github.com/damurka/atelier.git)
cd atelier
```

### 2. Set up a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

***(Note: If you are using Dash background callbacks, ensure you have diskcache or celery installed as per your configuration).***

## ⚙️ Configuration

Atelier requires API keys for maximum rate limits on external academic databases, as well as your chosen LLM provider.

1. Create a .env file in the root directory:

```bash
cp .env.example .env
```

2. Add your API keys to the .env file:

```.env
# Academic APIs
OPENALEX_API_KEY=your_openalex_key
SEMANTIC_SCHOLAR_API_KEY=your_s2_key
NCBI_API_KEY=your_pubmed_key

# LLM Provider
OPENAI_API_KEY=your_llm_key 

# App Settings
USER_AGENT="Atelier/0.1 (+[https://github.com/](https://github.com/)[your-username]/atelier; contact=you@example.com)"
```

## 💻 Usage

Start the Atelier server:

```bash
python app.py
```

1. Open your browser and navigate to http://localhost:8050.
2. Type a research question into the search bar (e.g., ***"What are the socioeconomic barriers to breast cancer screening in sub-Saharan Africa?"***).
3. Atelier will plan the search, fetch the papers, extract the metadata, and generate a fully cited synthesis summary.

## 🏗️ Architecture Overview

`pipeline/orchestrator.py`: The brain of the app. Manages the workflow from query generation to database saving.
`search/fetchers.py`: Object-oriented API wrappers utilizing a robust HttpClient.
`database/repository.py`: Abstracts Peewee ORM logic, handling hybrid text/vector local searches and chat history management.
`ui/`: Contains the Plotly Dash layouts, Tailwind styling components, and reactive callbacks.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See LICENSE for more information.