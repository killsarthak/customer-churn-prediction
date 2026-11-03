# Customer Churn Prediction 📉

Predicting which telecom customers are about to cancel their service ("churn"), so the business can step in **before** they leave. Keeping an existing customer is far cheaper than winning a new one, so even a rough early-warning model is genuinely valuable.

The project includes an interactive **Streamlit app** and a bundled logistic-regression model. It can score individual customers or CSV uploads immediately; upload a labeled dataset when you want to retrain the model and explore its portfolio-level evaluation. The serialized model contains the fitted pipeline and input metadata, not customer records.

---

## 📊 The Dataset

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — **7,043 customers** described by **21 columns**:

- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Target:** `Churn` (Yes / No)

About **26.5% of customers churn** and 73.5% stay — an *imbalanced* dataset, which shapes how we measure and train the models.

---

## 🗺️ Notebook Workflow

The notebook walks through a complete, beginner-friendly ML pipeline:

**1. Load & inspect** — read the data, check shapes, types, and the churn split.

**2. Data cleaning**
- `TotalCharges` was stored as text because of 11 blank entries (all customers with `tenure = 0`, i.e. brand new). Converted to numeric and filled those blanks with 0.
- Dropped `customerID` — a unique identifier carries no predictive signal and can cause a model to "memorise" rows.
- Encoded the target: `Churn` → 1 / 0.

**3. Exploratory Data Analysis (EDA)**
- **Tenure:** churn is highest in the first year (**~47%** in the first 12 months) and falls steadily to **~7%** after five years — the first year is the danger zone.
- **Contract:** month-to-month churns at **~43%** vs **~3%** for two-year contracts.
- **Internet:** fiber-optic customers churn far more (**~42%**) than DSL (**~19%**).
- **Interactions:** fiber **and** month-to-month together hit **~55%** churn — the riskiest segment. Adding tech support roughly halves fiber churn (**~49% → ~23%**).
- **Overlap check:** risky factors overlap (e.g. ~48% of month-to-month customers use electronic check), a reminder that *EDA shows association, not causation.*
- **Linear-model checks:** a correlation heatmap flags multicollinearity, and a log-odds plot confirms the numeric features are roughly linear in the log-odds of churn.

**4. Preprocessing**
- Collapsed redundant `"No internet service"` / `"No phone service"` categories into `"No"`.
- Dropped `TotalCharges` — it's essentially `tenure × MonthlyCharges`, so it's redundant (multicollinearity) and hurts coefficient interpretability.
- **Binary encoding** for Yes/No columns; **one-hot encoding** (with `drop_first`) for multi-category columns.
- **Train/test split** (80/20) and **standard scaling** (z-scores), fit on the training data only to avoid leakage.

**5. Handling class imbalance** — `class_weight="balanced"` (and `scale_pos_weight` for XGBoost) so the models pay extra attention to the rare churn class, boosting **recall**.

**6. Modeling** — three models, scored the same way at the default 0.50 cutoff:

| Model | Recall | Precision | ROC-AUC |
|-------|:------:|:---------:|:-------:|
| **Logistic Regression** | **0.820** | 0.523 | **0.860** |
| XGBoost | 0.796 | 0.534 | 0.855 |
| Random Forest | 0.458 | 0.643 | 0.839 |

Because a missed churner is a lost customer, we prioritise **recall**. Note how Random Forest's decent accuracy hides poor recall — accuracy alone is misleading on imbalanced data.

**7. Tuning**
- **Hyperparameters:** `GridSearchCV` with 5-fold cross-validation on ROC-AUC. Logistic Regression (best `C = 2`, CV ROC-AUC ≈ 0.839) and XGBoost (best `max_depth = 3`, `learning_rate = 0.03`, CV ROC-AUC ≈ 0.841) tied within a hair, so we keep the simpler, more interpretable model.
- **Threshold:** lowering the decision cutoff from 0.50 catches more churners. At **0.40** the model catches **327 of 373** churners (recall ≈ 0.88) while missing only 46 — a deliberate recall-vs-precision trade-off.

| Threshold | Recall | Precision | Caught | Missed |
|:---------:|:------:|:---------:|:------:|:------:|
| 0.50 | 0.820 | 0.523 | 306 | 67 |
| **0.40** | **0.877** | 0.466 | **327** | **46** |
| 0.30 | 0.941 | 0.428 | 351 | 22 |

**8. Save & reuse** — the final Logistic Regression model, the scaler, and the chosen threshold are saved together with `joblib`, then reloaded to score sample customers.

**9. Interpretability** — Logistic Regression coefficients are exponentiated into **odds ratios**, translating the model into plain business language (>1 raises churn odds, <1 lowers them). The top drivers line up with the EDA story.

---

## 🔑 Key Findings

From the odds ratios (>1 raises churn odds, <1 lowers them):

**Raises churn:** fiber-optic internet (**2.25×**) · streaming services · multiple lines · electronic-check payment · paperless billing.

**Lowers churn:** high monthly charges within a contract (**0.40×**) · longer tenure (**0.47×**) · two-year and one-year contracts (**0.53× / 0.75×**) · online security and tech support.

**Business takeaways:**
- Focus retention on customers in their first year.
- Nudge month-to-month customers toward longer contracts.
- Fiber customers without tech support churn heavily — bundling support could help.

---

## 🏆 Final Model

- **Notebook experiment:** Logistic Regression (`class_weight="balanced"`), selected for its balance of recall and interpretability.
- **Notebook decision threshold:** 0.40
- **Notebook test result:** approximately 88% recall (327 of 373 churners), trading some precision for that recall.

These figures describe the notebook experiment. The bundled `models/Model.pkl` is a separate logistic-regression pipeline trained from the Telco CSV, and the app also supports fitting and evaluating a fresh model on an uploaded dataset.

---

## 🚀 How to Run

### Run the Streamlit app locally

1. From the project folder, create an environment and install the app dependencies:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
3. Start the app:
   ```powershell
   streamlit run app.py
   ```
4. The bundled model is ready for single-customer and CSV scoring. To train and evaluate a new model from a labeled CSV, upload the Telco Customer Churn CSV in the sidebar and select **Train / refresh model**.

The training CSV needs the Telco feature columns and a `Churn` label (`Yes`/`No`). `customerID` and `TotalCharges` are optional; neither is used as a model feature. Missing numeric values are median-imputed, missing categories are imputed as `Unknown`, and unknown categories at scoring time are handled safely. Preprocessing is fitted only on the training split. Uploads are processed in the app session and are not written to the project.

### Regenerate the bundled model

The original dataset is not included. Download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as `data/raw/raw.csv`, then run:

```powershell
python src\train_model.py
```

The script saves `models/Model.pkl` and prints held-out test metrics. To use a CSV at another path, pass it as the first argument. The artifact includes the preprocessing pipeline, model, input choices, and metrics at supported thresholds; it does not include customer rows.

### Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository that you can access from [Streamlit Community Cloud](https://share.streamlit.io/).
2. In Streamlit Community Cloud, select **Create app**, choose the repository and branch, and set the app file to `app.py`.
3. Deploy. The platform installs the pinned-range dependencies from `requirements.txt`.
4. Open the deployed app to score customers with the bundled model, or upload a labeled CSV in the sidebar to train a fresh model.

No secrets or cloud credentials are required. The app does not bundle customer records.

### Run the notebook

The original notebook is still available for the full exploratory analysis and model comparison. Download the dataset and update the notebook's CSV path before running its cells. In the same virtual environment, install its additional libraries with `python -m pip install seaborn matplotlib xgboost joblib`.

Run the app tests with:

```powershell
python -m unittest discover -s tests -v
```

---

## 🧰 Built With

`Python` · `Streamlit` · `pandas` · `NumPy` · `scikit-learn` · `Plotly` · `XGBoost` · `seaborn` / `matplotlib` · `joblib`

---

## 📁 Repo Contents

```
customer_churn_prediction.ipynb   # the full annotated notebook
app.py                            # interactive Streamlit app
src/churn_model.py                # reusable preprocessing and logistic-regression pipeline
src/train_model.py                # train and save models/Model.pkl
models/Model.pkl                  # pre-trained pipeline and metadata
requirements.txt                  # app runtime dependencies
tests/test_app.py                 # focused app/model tests
README.md
```
