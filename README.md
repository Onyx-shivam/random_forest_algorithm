# random_forest_algorithm

# Market Outlook Predictor

A small Flask web app that serves a trained **Random Forest classifier** (`random_forest_algorithm.pkl`). You enter a company's layoff, hiring, AI and financial details, and the model predicts one of three outcomes:

| Class | Outcome |
|-------|---------|
| 0 | Bull Market |
| 1 | Recession |
| 2 | Sideways Market |

The page also shows how many of the trees agree with the result and a probability bar for each outcome.

> The model file does not store class names. The mapping above was inferred by testing the model (low layoffs and strong signals lead to class 0, very high layoffs and weak signals lead to class 1, in-between cases lead to class 2). If your training labels were encoded differently, edit the `OUTCOMES` list near the top of `app.py`.

---

## Project structure

```
your-folder/
├── app.py                        # Flask app + UI (single file)
├── requirements.txt              # Python dependencies
├── random_forest_algorithm.pkl   # Trained model (you provide this)
├── README.md
└── your_dataset.csv              # Optional: your training CSV, for real name matching
```

## Model details

- Type: `RandomForestClassifier` (scikit-learn 1.8.0), 50 trees
- Inputs: 21 features
- Output: 3 classes (0, 1, 2)

**Input features**

| Group | Features |
|-------|----------|
| Company | `company_name`, `industry`, `country`, `company_size` |
| Layoff event | `month`, `year`, `layoffs_count`, `layoff_percentage`, `reason_for_layoffs` |
| AI exposure | `ai_adoption_level`, `ai_automation_impact`, `ai_replacement_risk` |
| Hiring | `open_roles`, `hiring_trend`, `top_hiring_role`, `remote_jobs_percentage` |
| Financial health | `stock_growth_percent`, `revenue_growth_percent`, `salary_budget_change` |
| Employees | `employee_sentiment`, `job_security_score` |

## Requirements

- Python **3.11 or newer** (required by scikit-learn 1.8.0)
- The packages in `requirements.txt`: Flask, gunicorn, NumPy, pandas, scikit-learn 1.8.0

The scikit-learn version is pinned because the model was saved with it. Loading a pickle with a different version can fail or give wrong results.

## Run locally

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Open **http://localhost:5000** in your browser.

## Using the app

1. Fill in all the fields. Every field starts empty.
2. Select **Predict outcome**.
3. Read the result card (colour changes: green for Bull, red for Recession, amber for Sideways).
4. Select **Reset inputs** to clear the form.

**Number fields** (layoffs count, percentages, scores and so on) accept plain numbers. Commas are fine (`9,000`). A "Typical" range is shown under each field, taken from the values the model saw during training. Values outside the range are still accepted.

**Text fields** (company name, industry, country, company size, month, reason for layoffs, hiring trend, most-hired role) are free-typing boxes. The model only understands numbers, so each name has to be converted to the number used during training. See the next section.

## Real names vs. approximate codes

The `.pkl` file does not contain the name-to-number mapping from training. There are three ways the app handles a text field:

1. **Dataset CSV next to `app.py` (recommended).** Copy the CSV you trained on into the same folder and restart. The app reads the unique values of the eight text columns, sorts them A to Z (the way scikit-learn's `LabelEncoder` does) and uses them to match what you type. Matching ignores capitalisation. The terminal prints `Loaded real names from <file>.csv` when it works.
   - The CSV must contain all eight text columns with the same names as the features.
   - The number of unique values per column must match what the model was trained on: `company_name` 20, `industry` 7, `country` 6, `company_size` 4, `month` 12, `reason_for_layoffs` 5, `hiring_trend` 4, `top_hiring_role` 8. A column that doesn't match is skipped.
2. **Manual names.** Fill in the `MANUAL_LABELS` dictionary near the top of `app.py`, in the same A to Z order as your encoder:
   ```python
   MANUAL_LABELS = {
       "company_size": ["Large", "Medium", "Small", "Startup"],
   }
   ```
3. **Encoded number.** Typing the model's own code (for example `7` in a field) is treated as exact.

If a typed name is not found, the app still returns a result, but it uses a stable fallback number inside the valid range and shows a yellow notice that the result is only approximate. The same text always gives the same number.

If your categories were encoded with a custom dictionary rather than `LabelEncoder`, sorted order will not match. Use `MANUAL_LABELS` with your own order instead.

## API

The page talks to a small JSON API that you can call directly.

**`GET /health`** returns `{"status": "ok"}`.

**`POST /predict`** takes all 21 features as JSON (values can be strings or numbers):

```json
{
  "company_name": "Microsoft", "industry": "Technology", "country": "United States",
  "company_size": "Large", "month": "March", "year": "2025",
  "layoffs_count": "2400", "layoff_percentage": "9",
  "reason_for_layoffs": "AI automation", "ai_automation_impact": "6",
  "ai_replacement_risk": "6", "open_roles": "2300", "hiring_trend": "Growing",
  "remote_jobs_percentage": "48", "top_hiring_role": "Data Scientist",
  "stock_growth_percent": "24", "revenue_growth_percent": "15",
  "salary_budget_change": "7", "ai_adoption_level": "6",
  "employee_sentiment": "7", "job_security_score": "6"
}
```

Response:

```json
{
  "prediction": 0,
  "prediction_index": 0,
  "label": "Bull Market",
  "approximate": ["company name", "industry"],
  "probabilities": [
    {"class": 0, "label": "Bull Market", "key": "bull", "probability": 0.72, "note": "..."},
    {"class": 1, "label": "Recession", "key": "bear", "probability": 0.02, "note": "..."},
    {"class": 2, "label": "Sideways Market", "key": "flat", "probability": 0.26, "note": "..."}
  ]
}
```

`approximate` lists the fields whose names were not found. Missing or invalid inputs return HTTP 400 with an `error` message and a `fields` object naming each problem.

## Deploy on Render (optional)

1. Put `app.py`, `requirements.txt`, `random_forest_algorithm.pkl` (and your CSV if you use one) in a GitHub repository.
2. Create a new **Web Service** on Render from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Set the environment variable `PYTHON_VERSION` to a 3.11+ version (for example `3.12.8`), or add a `.python-version` file containing it.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` for the `.pkl` | Keep `random_forest_algorithm.pkl` in the same folder as `app.py`. |
| Install error for scikit-learn 1.8.0 | Your Python is older than 3.11. Upgrade Python, or install the scikit-learn version the model was trained with. |
| Model loads but warns about versions | The installed scikit-learn differs from the training version. Use the pinned version. |
| Yellow "approximate" notice | Names were not matched. Add your dataset CSV or fill `MANUAL_LABELS`. |
| Field turns red | It is empty or not a number. Fix the highlighted field. |
| Port 5000 already in use | Set another port: `PORT=5001 python app.py` (macOS / Linux) or `set PORT=5001 && python app.py` (Windows). |

## Disclaimer

This is a model estimate based on the data it was trained on. It is not financial or investment advice.