# Predicting African Inflation Risk: End-to-End ML Model, REST API & Mobile Application

## 📌 Project Overview & Mission
This project addresses the critical challenge of forecasting macro-financial instability and annual CPI inflation risk across African economies. By analyzing historical systemic, currency, sovereign-debt, and banking crisis indicators, I built a complete end-to-end machine learning pipeline—from exploratory data analysis and model training to a deployed FastAPI backend and a cross-platform Flutter mobile application.

The primary mission is to empower central bank analysts, international development researchers, and policy planners with an accessible, data-driven early-warning tool to identify periods of elevated inflation risk.

- **Kaggle Dataset Source**: [Africa Economic, Banking and Systemic Crisis Data](https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data)
- **Dataset Scope**: 1,059 historical observations across 13 African nations spanning 1860–2014 (`Algeria`, `Angola`, `Central African Republic`, `Egypt`, `Ivory Coast`, `Kenya`, `Mauritius`, `Morocco`, `Nigeria`, `South Africa`, `Tunisia`, `Zambia`, and `Zimbabwe`).
- **Features (11 Parameters)**: `country`, `year`, `exch_usd`, `systemic_crisis`, `domestic_debt_in_default`, `sovereign_external_debt_default`, `gdp_weighted_default`, `independence`, `currency_crises`, `inflation_crises`, and `banking_crisis`.

---

## Video Demo & Live Deployment Links
- **YouTube Video Demo**:  
  https://youtu.be/e-KE5x91Oa8  
  [Watch 7-Minute Technical & Application Walkthrough](https://youtu.be/e-KE5x91Oa8)
- **Live FastAPI Swagger UI (Interactive Testing)**:  
  https://linear-regression-model-3-1dmq.onrender.com/docs
- **Live Prediction POST Endpoint**:  
  https://linear-regression-model-3-1dmq.onrender.com/predict

> [!NOTE]
> Evaluators can test predictions directly in their browser using the **Swagger UI** link above or by launching the Flutter mobile application.

---

## 🧠 Machine Learning Development Narrative

### 1. Exploratory Data Analysis & Target Transformation
When I first inspected the raw dataset, the most striking pattern was the presence of extreme hyperinflation events—most notably Zimbabwe's 2008 hyperinflation, which reached nearly ~22,000,000% annual CPI growth. When computing initial correlation heatmaps on raw inflation data, virtually all crisis features showed near-zero correlation. The hyperinflation outlier was completely drowning out the learning signal for ordinary economic years.

To solve this, I engineered a **signed log1p transformation**:
$$\text{Transformed Target} = \text{sign}(x) \cdot \log(1 + |x|)$$

This transformation compressed extreme hyperinflation magnitudes by several orders of magnitude while preserving directional signs for deflationary years. Post-transformation, clear linear signals emerged: `inflation_crises` correlated at **0.54**, `year` at **0.39**, `currency_crises` at **0.35**, and `domestic_debt_in_default` at **0.28**.

### 2. Model Selection & Performance Evaluation
I evaluated four candidate regressors on an 80/20 train/test split:
1. **Linear Regression (Ordinary Least Squares)**
2. **SGDRegressor (Stochastic Gradient Descent)** — trained over 100 epochs using `partial_fit` to track empirical loss convergence.
3. **Decision Tree Regressor**
4. **Random Forest Regressor**

#### Loss Evaluation: Is the Loss High or Low?
The final test loss (Mean Squared Error) achieved by my winning **Random Forest model** is **low at ~0.052** (compared to SGD Linear Regression test MSE of **~0.168**). On the signed-log scale, an MSE of 0.052 translates to a tight prediction error margin of approximately $\pm 2\% - 4\%$ for typical economic years. 

#### Why Random Forest Won
Inflation risk in emerging markets is fundamentally non-linear—driven by compounding crisis factors (e.g. a currency crisis occurring simultaneously with a domestic debt default). Tree ensembles naturally capture these higher-order feature interactions better than linear hyperplanes.

#### How to Further Reduce Loss
To drive the prediction loss even lower in future iterations, I recommend:
1. **Adding External Macroeconomic Features**: Incorporating continuous time-series metrics such as broad money supply ($M_2$), central bank interest rates, and global oil/commodity price indices.
2. **Sequential Time-Series Modeling**: Using lagged features ($t-1, t-2$) or recurrent architectures (LSTM / ARIMA) to model multi-year inflationary momentum.
3. **Regional Stratification**: Training regional cluster models (e.g., ECOWAS vs. EAC trade blocs) to account for distinct monetary regimes.

---

## 🛠️ Hyperparameters & Model Optimization

### Understanding Hyperparameters
Hyperparameters are configuration settings specified *prior* to model training that govern learning capacity, tree architecture, and regularization. Unlike model parameters (weights/coefficients), which the algorithm learns automatically from data, hyperparameters must be tuned by the engineer.

### Key Hyperparameters Evaluated
- **Random Forest / Decision Tree**:
  - `n_estimators`: The total number of trees in the ensemble (e.g., 100 vs. 500 trees).
  - `max_depth`: Limits tree depth to prevent overfitting on historical noise.
  - `min_samples_split` & `min_samples_leaf`: Enforces minimum node population thresholds to ensure splits generalize well.
  - `max_features`: Restricts the feature subset evaluated per split ($\sqrt{p}$ vs. $\log_2(p)$).
- **SGDRegressor (Gradient Descent)**:
  - `learning_rate` & `eta0`: Controls step size updates (`constant`, `optimal`, `adaptive`).
  - `penalty` & `alpha`: Specifies regularization ($L_1$ Lasso, $L_2$ Ridge, or ElasticNet) and shrinkage magnitude ($\alpha$) to prevent exploding coefficients.

I used **GridSearchCV** with 5-Fold Cross-Validation to systematically select optimal hyperparameter combinations.

---

## ⚡ Production REST API Architecture (FastAPI)

To serve the model in production, I created a FastAPI application (`summative/API/prediction.py`):

1. **Robust Input Validation (Pydantic)**: Request payloads sent to `/predict` are validated against a strictly typed `PredictionInput` schema enforcing range bounds (e.g., `year` between 1860–2050, binary flag validation `0` or `1`, and valid country names). Invalid inputs return structured `422 Unprocessable Entity` HTTP responses.
2. **Artifact Pipeline & Reproducibility**: At server startup, pre-fitted artifacts (`best_model.joblib`, `scaler.joblib`, `banking_encoder.joblib`, `feature_names.json`, and `model_metadata.json`) load directly into memory, guaranteeing consistent preprocessing.
3. **CORS Middleware**: Explicitly configured via `CORSMiddleware` to support cross-origin requests from web browsers, Swagger UI, and mobile applications.

### Handling New Data & Model Retraining (`POST /retrain`)
Macroeconomic dynamics evolve over time due to policy changes or global shocks (*concept drift*). To handle new data:
- The API includes a live `POST /retrain` endpoint that ingests newly uploaded CSV datasets.
- When triggered, it automatically executes data cleaning, refits the scaler and encoders, retrains candidate regressors, and calculates updated Test MSE.
- If the newly trained candidate outperforms the current production model, it atomically overwrites the stored `.joblib` artifacts on disk in real-time without requiring an API server restart.

---

## 📱 Mobile Application Design (Flutter)

The mobile client (`summative/FlutterApp`) provides an intuitive frontend interface:

- **User-Friendly Form Layout**: Inputs are organized into clean fields, dropdown menus (`country`), and binary toggle switches for all 11 indicators.
- **One-Touch Sample Data Autofill**: A top header **Wand Icon** allows evaluators to populate representative test values with a single tap.
- **Dynamic API Base URL**: Users can switch seamlessly between local development (`http://127.0.0.1:8000`) and cloud production (`https://linear-regression-model-3-1dmq.onrender.com`).
- **Asynchronous Transport & Error Resilience**: Built using Flutter’s `http` package with non-blocking loading indicators and clear error banner alerts for network or validation failures.

---

## 🌍 Real-World Impact, Ethical Considerations & Limitations

- **Policy Value**: Early inflation risk forecasting provides economic decision-makers with actionable signals to adjust monetary policy before debt defaults and currency devaluation escalate.
- **Data Coverage Limitations**: The model is trained on 1,059 historical observations across **13 African nations** (1860–2014). Extrapolating predictions to non-represented African countries or post-2014 structural policy regimes requires careful human oversight.
- **Ethical Safeguards & Over-reliance Risks**: Machine learning forecasts should serve as supplementary early-warning tools alongside expert econometric judgment—never as single-source automated policy triggers.

---

## 🚀 How to Run the Project Locally

### 1. Run the FastAPI Backend
```bash
cd summative
.\.venv\Scripts\python.exe -m uvicorn API.prediction:app --reload --port 8000
```
*Access local Swagger UI documentation at `http://127.0.0.1:8000/docs`.*

### 2. Run the Flutter Mobile / Web App
```bash
cd summative/FlutterApp
flutter pub get
flutter run -d chrome     # Run in Chrome Browser
# OR
flutter run -d windows    # Run as Windows Desktop App
```
