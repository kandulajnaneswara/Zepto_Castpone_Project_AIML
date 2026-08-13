# Zepto Data & AI Platform

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-4B8BBE)](https://www.crummy.com/software/BeautifulSoup/)
![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white)
[![Joblib](https://img.shields.io/badge/Joblib-FF6F00?style=flat&logo=python&logoColor=white)](https://joblib.readthedocs.io/)
[![imbalanced-learn](https://img.shields.io/pypi/v/imbalanced-learn?label=imbalanced-learn&logo=python)](https://pypi.org/project/imbalanced-learn/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-used-blue?logo=huggingface)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

> **An end-to-end Data & AI platform demonstrating data engineering, analytics, machine learning, and grounded GenAI capabilities in one connected Zepto capstone project.**

## Project Overview

**Zepto Data & AI Platform** is an end-to-end AI/ML engineering capstone demonstrating how an AI/ML engineer can move across the full data and AI stack.

The repository contains three internally linked capabilities:

1. **Data Engineering — `/data_pipeline`**: scrapes raw book data, cleans and transforms it, stores it in a normalized SQLite database, executes SQL queries, and validates selected SQL results independently with Pandas.
2. **Analytics & Machine Learning — `/analytics`**: profiles and cleans the Titanic dataset, creates visual insights, builds and evaluates classification models, handles class imbalance, tunes Random Forest, performs a regression side-task, and saves the final pipeline.
3. **GenAI / RAG — `/support_assistant`**: embeds Zepto policy documents, routes questions through LangGraph, retrieves grounded context from ChromaDB, and exposes structured answers through FastAPI.

These are three capabilities within one coherent platform.

## Platform Architecture

```text
                         Zepto Data & AI Platform
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
       /data_pipeline         /analytics        /support_assistant
             |                    |                    |
       Web Scraping          EDA + Cleaning      Policy Documents
             |                    |                    |
       Data Cleaning         Visualization       Embeddings
             |                    |                    |
       SQLite + SQL          ML Modeling          ChromaDB
             |                    |                    |
       Pandas Validation     Evaluation           RAG Retrieval
                                  |                    |
                                  v                    v
                           Model Selection       FastAPI /ask
```

## Repository Structure

```text
zepto-data-ai-platform/
|
├── data_pipeline/
│   ├── data_pipeline.py
│   ├── books_toscrape.db
│   ├── output_file.csv
│   ├── Final_output_file.csv
│   ├── SQL_vs_Pandas.png
│   └── README.md
|
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic_best_pipeline.joblib
│   ├── README.md
│   └── output_files/
│       ├── titanic.csv
│       ├── titanic_clean.csv
│       └── plots/
│             ├── Age_distribution_by_Survival_outcome.png
│             ├── Correlation_Heatmap.png
│             ├── decision_tree_plot.png
│             ├── Fare_vs_Age_scatterplot.png
│             ├── pairplot.png
│             ├── residual_plot.png
│             ├── roc_curves.png
│             ├── standardization_before_after.png
│             ├── Survival_rate_by_pclass_and_sex.png
│             └── Univariate_analysis.png
|
├── support_assistant/
│   ├── docs/
│        ├── doc_01.txt
│        ├── doc_02.txt
│        ├── doc_03.txt
│        ├── doc_04.txt
│        ├── doc_05.txt
│        ├── doc_06.txt
│        ├── doc_07.txt
│        └── doc_08.txt
│   ├── ingest.py
│   ├── schemas.py
│   ├── prompt_template.py
│   ├── graph.py
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md
│   └── output_images
│       ├── FastAPI_UI.png                     
│       ├── Localhost_URL_JSON_response.png    
│       ├── query_1.png                        
│       ├── query_2.png                        
│       ├── query_3.png                        
│       └── Uvicorn_terminal_window.png                                
|
├── requirements.txt
└── README.md
```

> Generated artifacts such as the Support Assistant `chromadb/` vector store are created when the `\support_assistant` module is executed.

# Module 1 — Data Pipeline

## `/data_pipeline`

The Data Pipeline module covers scraping, cleaning, currency conversion, SQLite database loading, SQL querying, and Pandas verification.

### Pipeline Flow

```text
books.toscrape.com
        ↓
Web Scraping
        ↓
Raw Data Collection
        ↓
Data Cleaning & Type Conversion
        ↓
Currency Conversion
        ↓
SQLite Schema Creation
        ↓
Database Loading
        ↓
SQL Queries
        ↓
Pandas Verification
```

### Key Capabilities

- Scrapes books across the first 10 categories and follows pagination.
- Saves raw data to `output_file.csv`.
- Cleans and type-converts price, rating, and availability fields.
- Converts GBP to INR using the project-defined `1 GBP = 105.50 INR` baseline.
- Creates normalized `categories` and `books` SQLite tables.
- Executes `SELECT / WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, and `JOIN` queries.
- Includes a top-N-per-category query using `ROW_NUMBER() OVER (PARTITION BY ...)`.
- Reproduces a JOIN result using `pd.merge()` and compares it with the SQL result.

### Main Outputs

| Output | Purpose |
|---|---|
| `output_file.csv` | Raw scraped data |
| `Final_output_file.csv` | Cleaned data |
| `books_toscrape.db` | SQLite database |
| `SQL_vs_Pandas.png` | SQL vs. Pandas verification output |

[View the complete Data Pipeline README](data_pipeline/README.md)

# Module 2 — Analytics & Machine Learning

## `/analytics`

The Analytics module demonstrates the analyst-to-data-scientist workflow using the Titanic dataset.

### Pipeline Flow

```text
Titanic Dataset
      ↓
EDA & Profiling
      ↓
Missing-Value Handling
      ↓
Outlier & Skewness Analysis
      ↓
Visualization
      ↓
Stratified Train/Test Split
      ↓
Leakage-Safe Preprocessing
      ↓
Classification + Regression
      ↓
Evaluation
      ↓
Imbalance Handling
      ↓
Random Forest Tuning
      ↓
Saved Model Pipeline
```

### Key Capabilities

- EDA and data profiling.
- Missing-value handling.
- Univariate, bivariate, and multivariate analysis.
- Outlier and skewness analysis.
- Exploratory standardization check.
- Stratified 80/20 train/test split.
- `ColumnTransformer` and `Pipeline` preprocessing.
- Logistic Regression, Decision Tree, and Random Forest.
- Accuracy, Precision, Recall, F1, and AUC evaluation.
- Class-weight and SMOTE comparison.
- Random Forest `GridSearchCV` tuning.
- Fare regression with MAE, RMSE, R², and Adjusted R².
- Residual and heteroscedasticity analysis.
- Saved fitted pipeline using Joblib.

### Classification Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7697 | 0.6901 | 0.7206 | 0.7050 | 0.7541 |
| Random Forest | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |

The documented analysis identifies Random Forest as the strongest candidate among the three classifiers.

### Regression Results

| Model | MAE | RMSE | R² | Adjusted R² |
|---|---:|---:|---:|---:|
| Linear Regression (fare) | 21.14 | 41.75 | 0.3468 | 0.3118 |

### Visual Outputs

The module produces univariate analysis, correlation heatmap, survival-by-class-and-sex analysis, age distribution, fare-vs-age scatter plot, pair plot, standardization comparison, decision-tree visualization, ROC curves, and a regression residual plot.

[View the complete Analytics README](analytics/README.md)

# Module 3 — GenAI Support Assistant

## `/support_assistant`

The Support Assistant is an offline-gradable Retrieval-Augmented Generation service for Zepto customer support.

It uses Zepto policy documents as its knowledge source, routes incoming questions through LangGraph, retrieves relevant context from ChromaDB, and returns a structured response through FastAPI.

### RAG Flow

```text
Customer Query
      ↓
Intent Classification
      ↓
 ┌────┴──────────────┐
 ↓                   ↓
Policy Question   General Question
 ↓                   ↓
Embedding +       Direct Answer
ChromaDB Top-3
 ↓
Grounded Answer
 └───────┬───────────┘
         ↓
Pydantic Validation
         ↓
FastAPI POST /ask
```

### Key Capabilities

- Eight Zepto policy documents.
- Local embeddings with `all-MiniLM-L6-v2`.
- Persistent ChromaDB collection.
- Top-3 retrieval using cosine similarity.
- LangGraph `StateGraph` orchestration and conditional routing.
- Pydantic structured output validation.
- FastAPI `POST /ask` endpoint and Swagger UI.
- Deterministic `MOCK_LLM=1` graded baseline.
- Optional ungraded `MOCK_LLM=0` extension.
- Docker support.

### Structured Response

```json
{
  "answer": "string",
  "intent": "policy_question",
  "sources": ["doc_02", "doc_06", "doc_05"],
  "confidence": 1.0
}
```

### Run

```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --host 127.0.0.1 --port 7861
```

Open:

```text
http://127.0.0.1:7861/docs
```

[View the complete Support Assistant README](support_assistant/README.md)

# Technology Stack

| Area | Technologies |
|---|---|
| Data Engineering | Python, Requests/web scraping components, BeautifulSoup, Pandas, SQLite, SQL |
| Analytics & ML | Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, imbalanced-learn, Joblib |
| GenAI / RAG | Python, FastAPI, LangGraph, ChromaDB, Sentence Transformers, Pydantic, Docker |

# Installation

From the repository root:

```bash
pip install -r requirements.txt
```

Individual module README files contain the detailed execution instructions for each module.

# Running the Platform

## Data Pipeline

```bash
cd data_pipeline
python data_pipeline.py
```

## Analytics

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

## Support Assistant

```bash
cd support_assistant
python ingest.py
uvicorn main:app --host 127.0.0.1 --port 7861
```

Then open:

```text
http://127.0.0.1:7861/docs
```

# Project Outputs

### `/data_pipeline`

- Raw scraped CSV
- Cleaned CSV
- SQLite database
- SQL vs. Pandas verification output

### `/analytics`

- Raw and cleaned Titanic datasets
- EDA and visualization outputs
- Classification evaluation results
- ROC curves
- Decision tree visualization
- Regression residual plot
- Saved `titanic_best_pipeline.joblib`

### `/support_assistant`

- Embedded policy corpus
- Persistent ChromaDB vector store
- FastAPI service
- Structured JSON responses
- Swagger UI
- API execution screenshots
- Uvicorn terminal output

# Documentation

Each module contains a detailed README covering its implementation, execution steps, design decisions, outputs, visualizations, and validation.

- [`/data_pipeline/README.md`](data_pipeline/README.md)
- [`/analytics/README.md`](analytics/README.md)
- [`/support_assistant/README.md`](support_assistant/README.md)

# Reproducibility

The project documents module-specific reproducibility considerations:

- The Data Pipeline regenerates its SQLite schema when rerun.
- The Analytics module keeps raw and cleaned Titanic data separate and uses a leakage-safe preprocessing pipeline.
- The Analytics module saves the complete fitted preprocessing + classifier pipeline as `titanic_best_pipeline.joblib`.
- The Support Assistant regenerates its ChromaDB collection through `ingest.py`.
- The Support Assistant uses `MOCK_LLM=1` as the deterministic graded baseline.

# Submission Summary

**Zepto Data & AI Platform** demonstrates an end-to-end AI/ML engineering workflow within one repository:

> **Collect → Clean → Store → Analyze → Model → Evaluate → Retrieve → Generate → Serve**

The project brings together traditional data engineering, analytics and machine learning, and grounded GenAI into one coherent submission.
