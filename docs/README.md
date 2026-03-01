# Generative AI RAG Pipeline

This repository provides an enterprise‑grade retrieval‑augmented generation (RAG) pipeline that can plug into a variety of Large Language Model (LLM) providers.  The design embraces loose coupling between model clients, prompt templates, embedding generation and storage so that you can swap components without rewriting business logic.

## Features

- **Flexible model backends** – Choose between OpenAI, Anthropic and local Hugging Face models.  A simple factory picks the right client based on a YAML configuration file.
- **Prompt templates and chains** – Author complex instructions using Jinja2 templates and compose them into multi‑step reasoning chains.
- **Retrieval‑augmented generation** – Use embeddings to fetch relevant context from your document corpus and feed it into the LLM for grounded answers with source citations.
- **Modular preprocessing** – Clean, chunk and index raw documents through a configurable pipeline before feeding them into your vector store.
- **Batteries included** – Sample scripts are provided for environment setup, running tests, building embeddings and cleaning up caches.

## Directory Layout

```
generative_ai_project/
├── config/                # YAML files for model and logging configuration
├── data/                  # Runtime data (caches, embeddings, vector DB)
├── src/                   # Core application code
│   ├── core/              # LLM clients and model factory
│   ├── prompts/           # Prompt templates and chain helpers
│   ├── rag/               # Retrieval and generation logic
│   ├── processing/        # Data cleaning and chunking
│   └── inference/         # High level inference engine
├── docs/                  # Project documentation (this file)
├── scripts/               # CLI utilities for setup, testing and maintenance
├── .gitignore             # Files and directories to ignore in Git
├── Dockerfile             # Container definition for deployment
├── docker-compose.yml     # Multi‑service orchestration
├── pyproject.toml         # Packaging configuration
└── requirements.txt       # Python dependencies
```

## Getting Started

1. **Install dependencies**: Create a Python virtual environment and install the requirements.

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure providers**: Copy `config/model_config.yaml` and update the API keys or model paths.  Override values using environment variables in your shell or `.env` file.

3. **Preprocess your documents**: Use the scripts in the `scripts/` directory (e.g. `build_embeddings.py`) to clean, chunk and index your corpus.

4. **Run inference**: Initialise the `InferenceEngine` class from `src/inference/inference_engine.py` with your configuration and call `generate_answer()` to perform a retrieval‑augmented query.

## Extending the Project

The modular architecture makes it easy to extend support for new providers or features:

* **Add a new LLM provider** – Implement the `BaseLLM` interface in `src/core` and register it in `model_factory.py`.
* **Custom prompts** – Add Jinja2 templates to `src/prompts/templates.py` and define chains in `src/prompts/chains.py`.
* **Alternative vector stores** – Provide a new implementation of the `VectorStore` interface in `src/rag/vector_store.py`.

## License

This project is provided under the MIT License.  See the `LICENSE` file for details.
