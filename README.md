# Regression Analysis Mobile Application & API

## Mission & Problem Description
This project addresses the critical challenge of forecasting macro-financial instability and annual CPI inflation risk across African economies. By analyzing historical systemic, currency, sovereign-debt, and banking crisis indicators, our machine learning pipeline builds predictive models to deliver early inflation risk assessments. The mission is to empower decision-makers with accessible, data-driven inflation forecasting via a high-performance REST API and mobile app.

### Dataset Description & Source
- **Data Source**: [Africa Economic, Banking and Systemic Crisis Data (Kaggle)](https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data)
- **Overview**: Contains 1,059 historical observations across 13 African nations spanning 1860–2014 (`Algeria`, `Angola`, `Central African Republic`, `Egypt`, `Ivory Coast`, `Kenya`, `Mauritius`, `Morocco`, `Nigeria`, `South Africa`, `Tunisia`, `Zambia`, `Zimbabwe`).
- **Features**: Includes financial crisis indicators (`systemic_crisis`, `domestic_debt_in_default`, `sovereign_external_debt_default`, `currency_crises`, `inflation_crises`, `banking_crisis`), macroeconomic variables (`exch_usd`, `gdp_weighted_default`, `independence`, `year`), and target (`inflation_annual_cpi`).

### Key Data Visualizations & Interpretation
1. **Target Distribution (Raw CPI vs. Signed-Log Transform)**:
   - *Insight*: Raw CPI inflation data contains severe hyperinflation outliers (such as Zimbabwe's 2008 event of ~22,000,000%), which drown out linear regression gradients.
   - *Feature Engineering*: Applied a signed log1p transformation ($\text{sign}(x) \cdot \log(1 + |x|)$), compressing extreme values while preserving direction (handling deflation years safely).
2. **Correlation Heatmap (Post-Transformation)**:
   - *Insight*: Log transformation uncovered strong predictive signals: `inflation_crises` ($r = 0.54$), `year` ($r = 0.39$), `currency_crises` ($r = 0.35$), and `domestic_debt_in_default` ($r = 0.28$).
3. **Scatter Plot with Fitted Regression Line**:
   - Visualizes the linear decision boundary fit against key crisis features before and after SGD gradient descent training.

---

## Publicly Available API Endpoint (Swagger UI)
- **Public API Documentation & Interactive Browser Testing (Swagger UI)**:  
  [https://linear-regression-model-3-1dmq.onrender.com/docs](https://linear-regression-model-3-1dmq.onrender.com/docs)
- **Backend Prediction Endpoint (`POST`)**:  
  `https://linear-regression-model-3-1dmq.onrender.com/predict` *(Programmatic POST endpoint used by Flutter app & Swagger UI)*

> [!NOTE]
> Evaluators can test predictions directly using the **Swagger UI** (`/docs` endpoint). Enter the 11 input parameters (`country`, `year`, `exch_usd`, `systemic_crisis`, `domestic_debt_in_default`, `sovereign_external_debt_default`, `gdp_weighted_default`, `independence`, `currency_crises`, `inflation_crises`, `banking_crisis`) to receive instantaneous CPI inflation forecasts.

---

## Video Demo
- **YouTube Video Demo**: [Watch 7-Minute Demo Video](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) *(Replace `YOUR_VIDEO_ID` with your uploaded YouTube video link)*

### Video Demo Evaluation Questions & Responses

#### 1. Is your Loss High or Low, and what can you do to further reduce the loss based on the use case?
- **Current Loss Evaluation**: The test loss (Mean Squared Error) is **low**. On our log-transformed target ($\text{sign}(x) \cdot \log(1 + |x|)$), the winning Random Forest model achieved a **Test MSE of ~0.052** (compared to Stochastic Gradient Descent Linear Regression MSE of ~0.168). In untransformed inflation space, an MSE of 0.052 translates to a tight prediction error margin of roughly $\pm 2\% - 4\%$ for standard non-hyperinflationary economic years.
- **Strategies to Further Reduce Loss**:
  1. **Feature Expansion with External Macroeconomic Data**: Integrate additional economic indicators such as central bank interest rates, broad money supply ($M_2$), international oil/commodity price indices, and bilateral import/export tariffs.
  2. **Sequential Time-Series Modeling**: Incorporate temporal dynamics (e.g., lagged crisis indicators $t-1, t-2$, ARIMA, or LSTM recurrent architectures) to capture multi-year inflationary momentum rather than assuming independent annual observations.
  3. **Geographic Regional Clustering**: Sub-group models by trade blocs (e.g., ECOWAS, EAC, SADC) to allow model parameters to adapt to localized currency pegging and monetary policies.

---

#### 2. Are there things called Hyperparameters that can help improve the performance of your model?
- **Understanding Hyperparameters**: Yes! Hyperparameters are external configuration settings specified *prior* to training that govern the model's structural capacity and optimization process (unlike internal model parameters/weights learned automatically during training).
- **Key Model Hyperparameters**:
  - **Random Forest / Decision Trees**:
    - `n_estimators`: The number of trees in the forest ensemble (e.g., 100 vs. 500 trees).
    - `max_depth`: Controls tree depth to balance expressiveness and prevent overfitting on historical noise.
    - `min_samples_split` & `min_samples_leaf`: Enforces minimum node sizes to ensure splits generalize well.
    - `max_features`: Restricts feature subsets considered per split ($\sqrt{p}$ vs. $\log_2(p)$).
  - **SGDRegressor (Gradient Descent)**:
    - `learning_rate` & `eta0`: Controls step size per iteration (`constant`, `optimal`, `adaptive`).
    - `penalty` & `alpha`: Specifies regularization ($L_1$ Lasso, $L_2$ Ridge, or ElasticNet) and shrinkage strength ($\alpha$) to prevent coefficient explosion.
- **Optimization Strategy**: We tune these hyperparameters systematically using **GridSearchCV** or **RandomizedSearchCV** with 5-Fold Cross-Validation.

---

#### 3. What would happen if you had new data? How would you update your model performance?
- **Addressing Concept & Data Drift**: Macroeconomic dynamics change over time due to policy shifts, global supply chain disruptions, or global shocks. As new annual crisis and inflation data becomes available, existing models risk degradation due to *concept drift*.
- **Automated Production Retraining (`POST /retrain`)**:
  1. **API Endpoint Integration**: Our FastAPI backend includes a live `POST /retrain` endpoint designed to ingest newly uploaded CSV datasets.
  2. **Automated Pipeline Execution**: When triggered, the API automatically executes data cleaning, updates missing value handlers, refits the `StandardScaler` and `LabelEncoder`, and retrains all candidate regressors (OLS, SGD, Decision Tree, Random Forest).
  3. **Safe Model Artifact Swapping**: The pipeline computes Test MSE on the new dataset. If the newly trained candidate outperforms the current production model, it atomically overwrites `best_model.joblib`, `scaler.joblib`, and `model_metadata.json` on disk, updating production predictions instantaneously without requiring a server reboot.

---

## Task 4: Rubric Evaluation Questions & Detailed Answers

### Question 1: Model Selection, Feature Engineering & Optimization
**Question**: *Why was the specific target transformation chosen, how were model alternatives (linear vs non-linear) compared, and what justified selecting the final winning regressor?*

**Answer**:
- **Target Transformation**: Historical African inflation data contains extreme hyperinflation events (e.g., Zimbabwe in 2008 with ~22,000,000% annual CPI growth). Without preprocessing, standard Ordinary Least Squares (OLS) and Stochastic Gradient Descent (SGD) loss functions are completely dominated by these severe outliers, obscuring underlying relationships. We implemented a **signed log1p transformation** ($\text{sign}(x) \cdot \log(1 + |x|)$), which effectively compresses hyperinflation magnitudes while preserving non-negative/deflation directional dynamics.
- **Model Comparison**: We evaluated four distinct models on a standardized 80/20 train/test split:
  1. **Linear Regression (OLS)**: Provided baseline linear decision boundaries.
  2. **SGDRegressor (Gradient Descent)**: Trained iteratively over 100 epochs via `partial_fit` to monitor empirical loss convergence.
  3. **Decision Tree Regressor**: Evaluated non-linear threshold splits across financial crisis indicators.
  4. **Random Forest Regressor**: Ensembled decision trees to mitigate individual tree variance and model complex feature interactions.
- **Selection Rationale**: The **Random Forest Regressor** achieved the lowest Test Mean Squared Error (MSE ~0.052 vs SGD MSE ~0.168). Inflation risk in emerging markets is non-linear—driven by compounding crisis factors (such as simultaneous currency devaluation and domestic debt default). Tree ensembles naturally capture these higher-order feature interactions better than linear hyperplanes.

---

### Question 2: REST API Architecture, Validation & Maintainability
**Question**: *How is the FastAPI backend structured to handle data validation, CORS, pipeline reproducibility, and real-time model retraining?*

**Answer**:
- **Data Validation & Pydantic**: Inputs sent to the `POST /predict` endpoint are strictly validated against a Pydantic `PredictionInput` schema enforcing range bounds (e.g., `year` between 1860–2050, binary indicator values `0` or `1`, valid country selection from the 13 supported African nations). Invalid payload formats trigger descriptive `422 Unprocessable Entity` HTTP exceptions.
- **Artifact Pipeline & Reproducibility**: At API startup, pre-fitted artifacts (`best_model.joblib`, `scaler.joblib`, `banking_encoder.joblib`, `feature_names.json`, `model_metadata.json`) are dynamically loaded into memory. This guarantees that user inputs undergo identical feature ordering, label encoding, and standard scaling as applied during Task 1 training.
- **CORS & Middleware**: Configured via FastAPI `CORSMiddleware` to allow cross-origin requests from web frontends, Swagger UI interactive testing (`/docs`), and native mobile clients.
- **Model Retraining Endpoint**: The `POST /retrain` endpoint accepts newly uploaded CSV datasets, runs cleaning and feature engineering pipelines, retrains the candidate models, compares Test MSE, and safely overwrites production joblib artifacts on disk in real-time without requiring API restart.

---

### Question 3: Mobile Application UX Design & API Integration
**Question**: *How does the Flutter mobile application deliver a responsive user experience, handle error states, and integrate seamlessly with the deployed cloud backend?*

**Answer**:
- **Form Design & Autofill**: The mobile app presents a clean, structured UI organizing the 11 input parameters into intuitive input fields, dropdowns (`country`), and binary toggle switches. To streamline evaluation and manual testing, a header **Wand Icon (Sample Data Autofill)** populates representative test values with a single tap.
- **Dynamic Base URL Configuration**: Recognising that local development (`http://127.0.0.1:8000`) and cloud production (`https://linear-regression-model-3-1dmq.onrender.com`) use different network endpoints, the app exposes an editable API Base URL configuration field.
- **Asynchronous HTTP & Error Resilience**: Requests are dispatched asynchronously via Flutter's `http` package with non-blocking UI spinners during network transport. If the API returns validation errors, connectivity timeouts, or server errors, descriptive user-facing banner messages display exact status details rather than unhandled exception crashes.

---

### Question 4: Real-World Policy Relevance, Ethical Considerations & Model Limitations
**Question**: *What are the real-world policy implications of this model, and what ethical considerations or data limitations must decision-makers keep in mind?*

**Answer**:
- **Policy Utility**: Inflation forecasting enables central banks, international development agencies, and financial planners to evaluate macro-economic risk flags early, helping mitigate currency devaluation crises and debt default spirals before hyperinflation escalates.
- **Data & Geographic Coverage Limitations**: The underlying dataset spans 1,059 observations across **13 African nations** from 1860 to 2014. Predictions applied to non-represented African countries (out of 54 total) or post-2014 macroeconomic structural regimes must be interpreted with caution, as economic policy mechanisms and international trade dynamics have evolved significantly.
- **Ethical Considerations & Over-reliance Risks**: Automated ML predictions should serve as supplementary early-warning indicators alongside expert econometric judgment—never as single-source automated policy drivers. Automated economic decisions based solely on historical crisis indicators could inadvertently restrict emergency liquidity or trigger self-fulfilling sovereign credit rating downgrades if misapplied.


---

## Instructions to Run the Mobile Application

### Prerequisites
- **Flutter SDK**: `^3.12.0` or higher ([Install Flutter](https://docs.flutter.dev/get-started/install))
- **Python Backend**: Python `3.11+` (to run the local API server if testing locally)

---

### Step 1: Start the API Backend (Local Testing)
Navigate to the `summative` directory and start the Uvicorn server:
```bash
cd summative
.\.venv\Scripts\python.exe -m uvicorn API.prediction:app --reload --port 8000
```
*The API will be available locally at `http://127.0.0.1:8000/docs`.*

---

### Step 2: Run the Flutter Mobile / Web Application

1. **Navigate to the Flutter project folder**:
   ```bash
   cd summative/FlutterApp
   ```

2. **Install Flutter packages**:
   ```bash
   flutter pub get
   ```

3. **Launch the Application**:
   * **In Web (Chrome)**:
     ```bash
     flutter run -d chrome
     ```
   * **In Windows Desktop**:
     ```bash
     flutter run -d windows
     ```
   * **In Android Emulator / Device**:
     ```bash
     flutter run -d android
     ```

4. **Testing Predictions in the App**:
   * Once launched, tap the **Wand Icon (Fill Sample Data)** in the top right header to automatically populate test values across all 11 fields.
   * Click **Predict** to view the computed Annual Inflation percentage or error feedback.
   * If running against a deployed cloud API, update the **API Base URL** text field at the top of the app to your public URL (e.g., `https://linear-regression-model-3-1dmq.onrender.com`).
