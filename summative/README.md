# Regression Analysis Mobile Application & API

## Mission & Problem Description
This project addresses the critical challenge of forecasting macro-financial instability and annual CPI inflation risk across African economies. By analyzing historical systemic, currency, sovereign-debt, and banking crisis indicators, our machine learning pipeline builds predictive models to deliver early inflation risk assessments. The mission is to empower decision-makers with accessible, data-driven inflation forecasting via a high-performance REST API and mobile app.

---

## Publicly Available API Endpoint (Swagger UI)
- **Public API Documentation & Interactive Testing (Swagger UI)**:  
  `https://african-inflation-risk-api.onrender.com/docs` *(Replace with your deployed Render / Cloud API URL)*
- **Prediction Endpoint (`POST`)**:  
  `https://african-inflation-risk-api.onrender.com/predict`

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
   * If running against a deployed cloud API, update the **API Base URL** text field at the top of the app to your public URL (e.g., `https://african-inflation-risk-api.onrender.com`).
