# Autonomous AI Research Agent

An autonomous, end-to-end AI research agent capable of executing the complete academic research lifecycle. Built with **LangGraph** and **Gemini**, this system seamlessly orchestrates literature review, algorithmic implementation, experimental evaluation, and scientific writing in LaTeX.

## Features

- **End-to-End Autonomy**: Processes a research objective entirely from scratch, generating a complete, formatted `.tex` file with citations and results.
- **Agentic Workflow**: Employs LangGraph to manage complex state transitions and loop interactions across distinct research phases.
- **Automated Literature Review**: Integrates with arXiv and Wikipedia APIs to pull recent and foundational research, synthesizing insights tailored to the objective.
- **Dynamic Code Execution**: Writes, executes, and evaluates Python implementation code safely, utilizing error-correction loops.
- **Logging & Traceability**: Each run stores a complete, comprehensive trail in an automatically generated timestamped `logs/` directory.

## Architecture & Modules

The underlying workflow is directed by a core `main.py` state machine, moving through four specialized tool sets:

1. **Literature Module (`tools_literature.py`)**: Responsible for querying academic endpoints and generating comprehensive thematic literature reviews.
2. **Implementation Module (`tools_implementation.py`)**: Drafts and iterates upon machine learning / algorithmic scripts in Python (e.g., `implement_ssl_anomaly_detection.py`).
3. **Evaluation Module (`tools_evaluation.py`)**: Executes evaluation scripts (e.g., `evaluate_ssl.py`), parsing metrics or identifying crash traces to feedback into the implementation phase.
4. **Writing Module (`tools_writing.py`)**: Translates execution results and literature insights into a rigorously formatted LaTeX manuscript (`scientific_article.tex`).

## Getting Started

### Prerequisites

- Python 3.10+
- An API Key for Google Gemini (e.g., `GEMINI_API_KEY`)

### Installation

1. **Clone the repository:**
   ```bash
   # Add your git URL here if applicable
   git clone <repository_url>
   cd research_agent
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and ensure you configure the environment properly:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   ```

### Usage

To start a new autonomous research run, execute the `main.py` script:

```bash
python main.py
```

Prompts can be modified or injected via the system state inside the runner. The agent will initialize a new timestamped log directory.

## Logs Directory Note

The `logs/` directory contains outputs, trace histories, markdown drafts, and `.tex` outputs for each session. To ensure the repository remains fast and light, the `logs/` directory is **excluded from Git tracking**. If you need to share specific outputs, manually extract the relevant LaTeX or Markdown files.

## Acknowledgements

This pipeline represents a scalable framework structurally similar to leading automated research pipelines leveraging agentic RAG and self-reflection loops.
