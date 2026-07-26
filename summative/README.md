# Regression Analysis Mobile Application & API

## Mission & Problem Description
This project addresses the critical challenge of forecasting macro-financial instability and annual CPI inflation risk across African economies. By analyzing historical systemic, currency, sovereign-debt, and banking crisis indicators, our machine learning pipeline builds predictive models to deliver early inflation risk assessments. The mission is to empower decision-makers with accessible, data-driven inflation forecasting via a high-performance REST API and mobile app.

### Dataset Description & Source
- **Data Source**: [African Financial Crises Dataset (Kaggle)](https://www.kaggle.com/datasets/chirin/african-financial-crises)
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
- **Public API Documentation & Interactive Testing (Swagger UI)**:  
  `https://linear-regression-model-3-1dmq.onrender.com/docs`
- **Prediction Endpoint (`POST`)**:  
  `https://linear-regression-model-3-1dmq.onrender.com/predict`

> [!NOTE]
> Evaluators can test predictions directly using the **Swagger UI** (`/docs` endpoint). Enter the 11 input parameters (`country`, `year`, `exch_usd`, `systemic_crisis`, `domestic_debt_in_default`, `sovereign_external_debt_default`, `gdp_weighted_default`, `independence`, `currency_crises`, `inflation_crises`, `banking_crisis`) to receive instantaneous CPI inflation forecasts.

---

## Video Demo
- **YouTube Video Demo**: [Watch 7-Minute Demo Video](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) *(Replace `YOUR_VIDEO_ID` with your uploaded YouTube video link)*

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
