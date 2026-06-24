# BBBP Predictor & Drug Discovery Assistant

## Overview

This project combines machine learning and retrieval-augmented generation (RAG) to support blood-brain barrier permeability (BBBP) prediction and drug discovery workflows.

The system predicts whether a molecule is likely to cross the blood-brain barrier using molecular fingerprints, molecular descriptors, and an XGBoost classifier. In addition, a LangChain + FAISS based retrieval system provides domain-specific information related to BBB permeability and cheminformatics concepts.

## Features

* BBB permeability prediction from SMILES strings
* Molecular feature extraction using RDKit
* Morgan fingerprint generation
* Molecular descriptor calculation
* XGBoost classification model
* FastAPI deployment
* MLflow experiment tracking
* Retrieval-Augmented Generation (RAG)
* FAISS vector database
* HuggingFace sentence embeddings
* Domain-specific question answering endpoint

## Project Structure

```text
BBBP/
│
├── data/
│   └── bbbp_text.txt
│
├── faiss_index/
│
├── predictor.py
├── rag.py
├── main.py
├── bbbp_xgb.pkl
├── requirements.txt
└── README.md
```

## Machine Learning Pipeline

```text
SMILES
   ↓
RDKit Feature Extraction
   ↓
Morgan Fingerprints + Molecular Descriptors
   ↓
XGBoost Classifier
   ↓
BBB Permeability Prediction
```

## RAG Pipeline

```text
Domain Knowledge
    ↓
Chunking
    ↓
Sentence Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Context Retrieval
```

## API Endpoints

### Predict BBB Permeability

POST `/Upload_smile`

Example Request:

```json
{
  "smile": "CCO"
}
```

### Ask Domain Questions

POST `/ask`

Example Request:

```json
{
  "question": "What are Morgan fingerprints?"
}
```

## Technologies Used

* Python
* FastAPI
* XGBoost
* RDKit
* Scikit-Learn
* MLflow
* LangChain
* FAISS
* HuggingFace Embeddings
* Sentence Transformers

## Future Work

* Integrate local LLMs for answer generation
* Support PDF ingestion and document-based RAG
* Molecular explainability using SHAP
* Model monitoring and experiment comparison dashboards


