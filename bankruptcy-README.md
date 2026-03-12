# 🏦 Bankruptcy Prediction Model — Investment Banking Risk System

An end-to-end machine learning system predicting corporate bankruptcy risk for an investment banking firm using **64 financial ratios from 31,000+ Polish companies (2000–2013)**. Built with DataRobot AutoML, the final XGBoost model achieved **AUC-ROC of 0.9858**, providing a **4–6 month early warning lead** over traditional detection methods with a modeled **$42–70M annual portfolio value**.

> **WPI MIS 587: Business Applications in Machine Learning — Team 10**

---

## 📊 Results at a Glance

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.9858** |
| **F1 Score** | 0.8939 |
| **Precision** | 0.9432 |
| **Recall (Sensitivity)** | 0.8494 |
| **LogLoss (Validation)** | 0.1125 |
| **LogLoss (Cross-Validation)** | 0.1178 |
| **LogLoss (Holdout)** | 0.1192 |

---

## 🏆 Model Selection — DataRobot AutoML Leaderboard

| Model | Validation LogLoss | Cross-Val LogLoss | Holdout LogLoss |
|---|---|---|---|
| **XGBoost (Early Stopping)** ✅ | **0.1125** | **0.1178** | **0.1192** |
| XGBoost (Standard) | 0.1136 | 0.1288 | 0.1296 |
| LightGBM (Early Stopping) | 0.1201 | 0.1341 | 0.1344 |
| LightGBM on ElasticNet | 0.1271 | 0.1472 | 0.1567 |
| Random Forest | 0.1672 | 0.1739 | 0.1719 |

> XGBoost selected for its lowest LogLoss across all three partitions, strong AUC, built-in L1/L2 regularization, and class imbalance handling via internal class-weighting strategies.

---

## 🗂️ Dataset

- **Source:** [UCI Machine Learning Repository — Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- **Size:** 31,000+ companies across 5 yearly datasets (2000–2013)
- **Features:** 64 financial ratio features — profitability, liquidity, leverage, operational efficiency
- **Target Variable:** Binary bankruptcy status (1 = bankrupt, 0 = operating)
- **Class Imbalance:** ~20% bankrupt, ~80% operating

---

## 🔧 Project Workflow

### 1. Data Preparation & EDA (`baml-1.ipynb`)
- Combined 5 yearly datasets into a unified training set
- Identified significant class imbalance (~20% bankruptcy rate)
- Correlation analysis: top positively correlated features with bankruptcy (debt ratios, liabilities)
- Top negatively correlated features (company size via log total assets, working capital)
- Spearman hierarchical clustering to identify redundant feature groups
- Mutual information scoring and Random Forest feature importance for selection

### 2. Feature Engineering (`baml-2.ipynb`)
- **Missing values:** Dropped X37 (42% missing); imputed X21 (Sales Growth) with median + missingness flag
- **Outliers:** Winsorized 4 heavily right-skewed features (X20, X43, X44, X58) at 95th percentile
- **Skewness:** Log/inverse transformations for left-skewed features (e.g., X56 gross margin ratio)
- **Composite features engineered:**
  - `Altman Z-Score` — working capital/total assets, retained earnings/total assets, EBIT/total assets, equity/liabilities, sales/total assets
  - `Liquidity Stress Score` — average of liquidity ratios (X3, X5, X12)
  - `Inventory Leverage Risk` — winsorized inventory turnover × current ratio
  - `Margin Cost Squeeze` — gross margin × winsorized payables days
- **Binary risk flags:**
  - `Liquidity Crisis Flag` (X4 < 1)
  - `Negative Equity Flag` (X10 < 0)
  - `Interest Coverage Risk Flag` (X27 < 1)

### 3. Model Training (DataRobot AutoML)
- Evaluated 10+ models including Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks, GBT, and XGBoost
- Primary metric: LogLoss (chosen over accuracy due to class imbalance)
- Secondary metric: AUC-ROC
- XGBoost with Early Stopping selected as final model
- Threshold tuned to prioritize recall (catching more true bankruptcies) over precision

### 4. Model Explainability (SHAP)
Top 5 bankruptcy drivers identified via SHAP feature importance:

| Feature | Description | Impact |
|---|---|---|
| **X27** | Interest Coverage Ratio | Low coverage → highest bankruptcy risk |
| **X34** | Accounts Payable Days | Extended payables → liquidity stress |
| **X46** | Working Capital Turnover | Low efficiency → higher risk |
| **X58_winsorized** | Payables Payment Period | Extended periods → cash flow issues |
| **X6** | Retained Earnings / Total Assets | Higher retained earnings → protection |

---

## 💼 Business Impact

### Early Warning System
- Provides **4–6 month lead time** over traditional bankruptcy detection
- Enables proactive portfolio rebalancing before distress materializes

### Tiered Portfolio Value

| Client Tier | Market Cap | Annual Savings | Implementation Cost | ROI |
|---|---|---|---|---|
| Tier 1 (Large Enterprise) | >$10B | $25–40M per 10 companies | $3–5M | 400–700% |
| Tier 2 (Mid-Market) | $1–10B | $12–20M per 20 companies | $1.5–2.5M | 380–700% |
| Tier 3 (Small/Growth) | <$1B | $5–10M per 50 companies | $1–2M | 150–400% |

**Total modeled portfolio value: $42–70M annually**

### Risk Segmentation
- **High-risk (>0.7):** Urgent portfolio rebalancing + distressed asset opportunities
- **Medium-risk (0.3–0.7):** Increased monitoring + targeted advisory services
- **Low-risk (<0.3):** Standard protocols + opportunity analysis

---

## 🧰 Tech Stack

```
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · DataRobot AutoML · SHAP
```

**Techniques:** XGBoost · AutoML · SHAP Explainability · Winsorization · Feature Engineering · Class Imbalance Handling · Threshold Optimization · Partial Dependence Plots

---

## 📁 Repository Structure

```
├── baml-1.ipynb          # EDA, correlation analysis, feature selection
├── baml-2.ipynb          # Feature engineering pipeline
├── README.md
```

> Note: Final model training was performed on DataRobot AutoML platform. EDA and feature engineering notebooks are included above.

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/srikrishna-poduri/bankruptcy-prediction-ml.git
cd bankruptcy-prediction-ml

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# 3. Download dataset from UCI ML Repository (link above)

# 4. Run notebooks in order
jupyter notebook baml-1.ipynb   # EDA & feature selection
jupyter notebook baml-2.ipynb   # Feature engineering
```

---

## 👥 Team

**Team 10 — WPI MIS 587**
- Sri Krishna Datta Poduri
- Pradyumn Tendulkar
- Sagar Hegde
- Madhava Kalyan Gadiputi
- Rama Raju Kanumuri

---

## 📚 References

- Zięba et al. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. *Expert Systems With Applications*, 58, 93–101.
- UCI ML Repository — [Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)

---

**Sri Krishna Datta Poduri** | MS Data Science @ WPI
[LinkedIn](https://www.linkedin.com/in/manikyalaraopoduri) · [GitHub](https://github.com/srikrishna-poduri)
