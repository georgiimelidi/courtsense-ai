# CourtSense AI

Machine learning system for tennis match prediction and tournament simulation using XGBoost, Elo ratings, historical retrieval, and contextual analysis.

---

## Demo

### Match Predictor

<p align="center">
  <img src="demo_match_pred.jpeg" width="900">
</p>

### Roland-Garros 2026 Simulation

<p align="center">
  <img src="RG26_prediction.jpeg" width="900">
</p>

---

## Overview

CourtSense AI predicts ATP tennis matches using:

- XGBoost classification
- Dynamic Elo ratings
- Surface-specific player strength
- Historical match retrieval
- Head-to-head analysis
- Contextual reasoning agents
- Roland-Garros tournament simulation

The project combines probabilistic prediction with interpretable reasoning layers.

---

## Main Features

### Match Prediction

Predicts:
- match winner probability
- confidence level
- contextual evidence

using:
- ATP rankings
- ranking points
- Elo ratings
- surface-specific Elo
- recent form
- fatigue indicators
- serve/return profile

---

### Roland-Garros 26 Simulation

Uses the official Roland-Garros 2026 draw to simulate:
- projected winners
- round progression
- tournament champion

---

### Retrieval System

Retrieves historically similar matches using:
- cosine similarity
- normalized feature vectors
- contextual match-state similarity

---

### Contextual Agent Analysis

Specialized agents analyze:
- head-to-head history
- fatigue
- matchup dynamics
- similar historical matches
- news context
- uncertainty risk

The final prediction combines:
- ML probability
- contextual evidence
- agent reasoning

---

## Model Architecture

```text
Historical ATP data
        ↓
Feature engineering
        ↓
Elo computation
        ↓
XGBoost prediction
        ↓
Historical retrieval
        ↓
Agent analysis
        ↓
Final judgement
```

---

## Core Features Used by XGBoost

- `overall_elo_diff`
- `surface_elo_diff`
- `rank_diff`
- `rank_points_diff`
- `recent_win_rate_diff`
- `age_diff`
- surface encoding
- best-of format

---

## Elo System

The project maintains:
- overall Elo
- clay Elo
- hard Elo
- grass Elo

Each rating is updated dynamically from historical ATP matches.

---

## Project Structure

```text
courtsense-ai/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── agents.py
│   ├── draw_loader.py
│   ├── elo.py
│   ├── explainability.py
│   ├── features.py
│   ├── head_to_head.py
│   ├── model.py
│   ├── news_context.py
│   ├── player_stats.py
│   ├── predict.py
│   ├── retrieval.py
│   └── tournament_simulator.py
│
├── data/
│
├── model.pkl
│
└── README.md
```

---

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

---

## Current Performance

| Metric | Value |
|---|---|
| Accuracy | ~66% |
| ROC-AUC | ~0.73 |
| Log Loss | ~0.61 |

---

## Future Improvements

- bookmaker odds integration
- richer retrieval system
- improved fatigue modeling
- probabilistic bracket simulation

---

<p align="center">
Built by <b>Georgii Melidi</b> · 2026
</p>
