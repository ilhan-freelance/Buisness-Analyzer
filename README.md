# 🚨 Retail Crash-Test Engine

> **"Don't just analyze your sales — stress-test your survival."**

---

## 🎯 Problem Statement

Most retail dashboards are rear-view mirrors — they show you what *happened*, not what *could* happen. When a competitor slashes prices, a supply chain breaks, or foot traffic collapses, traditional analytics leave you blind. The Retail Crash-Test Engine simulates real-world market shocks and calculates your business fragility *before* disaster strikes, so you can act instead of react.

---

## 💡 Unique Business Angle

This is **not** a basic sales dashboard. The Retail Crash-Test Engine is a **Business Stress-Test Simulator** built around three core pillars:

- **Business Fragility Score** — A single 0–100 composite score quantifying how exposed your retail operation is to demand shocks, revenue concentration risk, and customer churn.
- **Live Disaster Simulator** — Apply three configurable shock scenarios (e.g. price war, demand collapse, high-value customer exodus) and see projected revenue impact in real time.
- **Auto-Generated Survival Shield** — Rule-based, AI-free survival recommendations tailored to your fragility profile and cluster composition — no LLM API required, fully offline-capable.

---

## 🧠 ML Methodology

| Component | Detail |
|-----------|--------|
| **Algorithm** | K-Means Clustering (k = 3 to 5, default k = 4) |
| **Features** | Age, Quantity, Price per Unit, Month, DayOfWeek |
| **Data Leakage Prevention** | `Total Amount` is **explicitly excluded** — it is a linear combination of `Quantity × Price per Unit`, and including it would leak target-correlated information into the feature space |
| **Scaling** | `StandardScaler` (zero mean, unit variance) |
| **Evaluation** | Silhouette Score + Elbow Method |

**Output — 4 Behavioral Customer Personas:**

1. 👑 **Core VIP Pillars** — High frequency, high spend; your revenue backbone
2. 💸 **Price-Sensitive Bargain Hunters** — Volume buyers who chase discounts
3. 🎯 **High-Ticket Occasional Buyers** — Infrequent but high-value transactions
4. ⚠️ **At-Risk / Low-Frequency** — Disengaged customers at churn risk

---

## 📊 Business Intelligence Features

1. **Business Fragility Score (0–100)** — Composite metric measuring revenue concentration, cluster health, and demand volatility
2. **Live Disaster Simulator (3 shock scenarios)** — Model the revenue impact of price wars, demand drops, or VIP churn
3. **Auto-Generated Survival Shield (rule-based, no LLM API)** — Actionable mitigation strategies generated from your data profile
4. **CEO Crash-Test Report (downloadable `.md`)** — One-click export of your full stress-test results for stakeholder reporting

---

## 🗂️ Project Structure

```
├── app.py                    # Main Streamlit application
├── sample_retail_data.csv    # Synthetic retail dataset (1,500 rows)
├── analysis.ipynb            # EDA + ML Notebook
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Run Locally

### Option 1: Zero-Setup Instant Browser Preview (No Python Required)
Simply double-click `index.html` or open it in any web browser:
```bash
start index.html
```
Full client-side ML clustering, Disaster Simulator, and Fragility Score run instantly inside your browser!

### Option 2: Python / Streamlit Local Server
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment Options

### A. Deploy to Vercel (1-Click, Zero Config)
1. Push this repository to GitHub
2. Import project in [vercel.com](https://vercel.com/new)
3. Framework Preset: **Other** (Root directory has `index.html` + `vercel.json`)
4. Click **Deploy** — your live app URL is generated in under 15 seconds!

### B. Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Click **Deploy**

---

## 📋 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| Transaction ID | String | Unique transaction identifier |
| Date | Date | Transaction date |
| Customer ID | String | Customer identifier |
| Gender | String | Male / Female |
| Age | Integer | Customer age (18–65) |
| Product Category | String | Electronics / Clothing / Beauty / Home & Kitchen |
| Quantity | Integer | Units purchased (1–5) |
| Price per Unit | Float | Price per item |
| Total Amount | Float | Quantity × Price per Unit |

---

## ⚠️ Assumptions & Limitations

- Clustering is unsupervised; customer personas are assigned **heuristically** from cluster centroids and may require manual review for domain alignment
- The sample dataset is **synthetic**; results on real-world retail data will vary
- The Disaster Simulator uses **linear scaling models** for revenue impact estimation — non-linear effects (e.g. panic buying, brand damage) are not modelled

---

## 🏆 Hackathon

| | |
|---|---|
| **Topic** | Retail Sales & Customer Behaviour Analysis |
| **Built for** | College Hackathon 2026 |
