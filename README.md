# Spaceship Titanic Classification — Kaggle Top 6% Solution

A competitive machine learning solution achieving a Top 6% standing in Kaggle's Spaceship Titanic competition by engineering domain-specific features and benchmarking gradient-boosted tree models.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/spaceship-titanic-top-6-for-beginners)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Classification%20/%20Competitive%20ML-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Spaceship Titanic Classification — Kaggle Top 6% Solution](https://www.kaggle.com/code/lazer999/spaceship-titanic-top-6-for-beginners)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Achieved **Top 6% finish** on the Spaceship Titanic global leaderboard.
- Extracted high-signal passenger features from raw identifiers (Cabin Deck, Num, Side, Group ID, Family Size).
- Engineered total onboard luxury spending aggregates (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck).
- Benchmarked and tuned multiple algorithms: CatBoost Classifier, XGBoost, and Gradient Boosting.
- Designed clean visualization workflows highlighting transport probabilities across demographic slices.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[Passenger Manifest] --> B[Feature Extraction: Cabin, Groups, Spending]
    B --> C[Iterative Imputation & Encoding]
    C --> D[Model Benchmarking: CatBoost vs XGBoost]
    D --> E[Hyperparameter Tuning]
    E --> F[Top 6% Ensemble Predictions]
```

---

## Repository Structure

```plaintext
spaceship-titanic-top-6-percent/
├── notebooks/
│   └── spaceship-titanic-top-6-percent.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/spaceship-titanic-top-6-percent.git
cd spaceship-titanic-top-6-percent
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/spaceship-titanic-top-6-percent.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Spaceship Titanic Competition Dataset](https://www.kaggle.com/c/spaceship-titanic)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle competitions download -c spaceship-titanic
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Spaceship Titanic Classification — Kaggle Top 6% Solution](https://www.kaggle.com/code/lazer999/spaceship-titanic-top-6-for-beginners)

If you found this project helpful or insightful, please consider starring the repository ⭐!
