# IPL Dashboard – Random Forest and Linear/Logistic Regression Edition

## What is added
The working IPL Streamlit dashboard now supports an algorithm dropdown on model-based prediction pages:

- **Random Forest**
  - Numerical prediction: `RandomForestRegressor`
  - Classification prediction: `RandomForestClassifier`

- **Linear / Logistic Regression**
  - Numerical prediction: `LinearRegression`
  - Classification prediction: `LogisticRegression`

## Pages with model selection
- ML - Batter vs Bowler Prediction
- ML - Match Winner Prediction
- ML - Bowler Runs Prediction
- ML - Batter Runs Prediction
- ML - Advanced Team Winner Prediction

## Pages intentionally unchanged
- **ML - Orange & Purple Cap Prediction** keeps the corrected fixture-aware baseline method:
  `actual recorded standings + projected additions from selected remaining fixtures`.
  This avoids re-predicting the entire standings unrealistically.
- **ML - Full Match Scorecard Simulation** keeps its T20 simulation engine and Impact Player scenario logic.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files
- `app.py` – complete updated Streamlit application
- `requirements.txt` – dependency list
- `README.md` – this usage note
