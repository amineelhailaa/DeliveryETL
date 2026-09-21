# DeliveryETA — Five-day project roadmap

Build a machine learning application that answers: **“How many minutes will this delivery take?”**

Follow this README in order. Check each task only when its completion criteria are met. It is currently a work plan; replace the pending dataset information, results, and screenshots with your actual findings before submission.

**Project period:** Monday, September 21 to Friday, September 25, 2026.  
**Official deadline:** September 25, 2026, before midnight.  
**Personal submission target:** Friday at 18:00, leaving time to fix submission problems.  
**Planning assumption:** approximately 6 focused hours per day, 30 hours total. Breaks are additional. Adjust working hours to your availability while preserving the order and Friday’s verification time.

## 1. What you must deliver

- [ ] An organized Git repository with `data/`, `notebooks/`, `src/`, `models/`, and `app/`.
- [ ] A complete, commented notebook covering cleaning, EDA, feature engineering, modeling, and evaluation.
- [ ] At least **two new features**, with explanations.
- [ ] At least **four regression models**, compared fairly.
- [ ] An **80% training / 20% test** split.
- [ ] A scikit-learn pipeline containing preprocessing and the estimator, with no training/test leakage.
- [ ] Hyperparameter optimization of the selected model.
- [ ] MAE, MSE, RMSE, R², and adjusted R², with train/test comparison and business interpretation.
- [ ] A saved trained model/pipeline and fitted scaler, using `joblib` or `pickle`.
- [ ] A Streamlit application with input fields, a prediction in minutes, data visualizations, and model performance visualizations.
- [ ] A `requirements.txt` and verified installation/run instructions.
- [ ] A final README describing the dataset, approach, results, limitations, and application screenshots.
- [ ] A working Jira board link.
- [ ] Preparation for a 30-minute assessment: 10-minute demonstration, 10-minute code review, and 10-minute practical exercise.

**Optional, only after everything above works:** FastAPI, Apache Airflow, MLflow. No bonus is needed to finish the required scope.

## 2. The daily schedule

| Day | Focus | Time budget | Required checkpoint |
| --- | --- | --- | --- |
| Mon Sep 21 | Setup, data audit, cleaning, splitcd | 6 h | Data issues documented; reproducible cleaning and split |
| Tue Sep 22 | EDA, features, preprocessing, baseline | 6 h | EDA complete; two features; working baseline pipeline |
| Wed Sep 23 | Four models, comparison, tuning | 6 h | Model selected and tuned using training data only |
| Thu Sep 24 | Final evaluation, saved artifacts, Streamlit | 6 h | Application loads the saved pipeline and predicts |
| Fri Sep 25 | Verification, documentation, delivery, rehearsal | 6 h | Repository and Jira ready; submitted before the deadline |

At the end of each day: update Jira, save your work, commit a coherent milestone, and write tomorrow’s first action. Leave evidence in the notebook or repository, rather than relying on memory.

## 3. Planned repository structure

The structure below is the project target. Some files currently exist as empty placeholders and still need implementation and verification.

```text
DeliveryETA/
├── README.md
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── data/
│   ├── raw/                         # Original Food delivery.csv; keep unchanged
│   └── processed/                   # Reproducible cleaned data and split metadata
├── notebooks/
│   └── 01_delivery_eta.ipynb        # Complete, commented analysis
├── src/
│   ├── __init__.py
│   ├── data.py                     # Loading and deterministic cleaning
│   ├── features.py                 # Shared feature calculations
│   ├── train.py                    # Split, pipelines, CV, tuning, serialization
│   ├── evaluate.py                 # Metrics and diagnostic plots
│   └── predict.py                  # Load pipeline and predict from raw inputs
├── models/
│   ├── delivery_eta_pipeline.joblib
│   ├── scaler.joblib
│   └── metadata.json               # Features, versions, seed, selected parameters
├── app/
│   └── streamlit_app.py
└── reports/
    ├── figures/
    ├── screenshots/
    ├── cleaning_log.md
    ├── model_comparison.csv
    └── final_metrics.json
```

Keep the project small. Move reusable code into `src/`; use the notebook to explain decisions and display results. The notebook, training process, and application must use the same feature calculations.

## 4. Monday — Understand and clean the data

### ETA-01 · Set up the project, Docker, and Jira — 60 min

- [x] Create the folders above and initialize Git if necessary.
- [ ] Keep the Python dependencies in `requirements.txt`; Docker will install pandas, NumPy, matplotlib, seaborn, scikit-learn, SciPy, joblib, Streamlit, and Jupyter from this file.
- [ ] Create a `Dockerfile` based on a fixed Python version, set a working directory, copy and install `requirements.txt`, then copy the project code.
- [ ] Configure `compose.yaml` with one reusable application service. Mount the project folder for development and expose port `8501` for Streamlit and `8888` for Jupyter when needed.
- [ ] Add `.git`, `.idea`, Python caches, notebook checkpoints, raw data that cannot be redistributed, and secrets to `.dockerignore`. Add `.idea/`, caches, checkpoints, and secrets to `.gitignore`. Keep the required trained artifacts available for delivery.
- [ ] Build the image with `docker compose build`, then prove that Python can import the required libraries inside the container.
- [ ] Create a Jira board with **To do → In progress → Done**. Copy the `ETA-01` to `ETA-20` tasks from this README into issues; actual Jira issue keys may differ.
- [ ] Add each issue’s estimate, planned date, and completion criteria. Put the board link in this README.

**Done when:** the image builds from a clean Docker cache, the notebook or Python process runs inside the container, imports work, and the project tasks are visible in Jira.

### ETA-02 · Load and audit the dataset — 75 min

- [x] Put the supplied dataset in `data/raw/` (`dataset-1.csv` is currently present). Keep this raw file unchanged; rename it only if you also update every documented path.
- [ ] Record its source, row count, column count, column names, types, sample values, and missing-value counts.
- [ ] Identify the target `Time_taken(min)` and verify its unit and definition against any available dataset documentation.
- [ ] Check whether the target actually measures time **from order to receipt**. If the start point is undocumented or different, record that limitation rather than claiming otherwise.
- [ ] Create a small data dictionary: column, meaning, type, unit, and whether it is available when the prediction is made.
- [ ] Identify delivery/order identifiers, courier identifiers, timestamps, and suspicious fields that could reveal the outcome.

**Done when:** you can explain what one row represents, what the target measures, and which inputs are available at prediction time.

### ETA-03 · Implement and document cleaning — 150 min

- [ ] Trim whitespace and standardize category spelling and capitalization.
- [ ] Inspect placeholder values such as empty strings, `NaN`, and `None`; convert actual missing-value markers to nulls without turning valid categories into missing data.
- [ ] Inspect text wrappers in numerical or categorical fields. Extract real values only where the observed format requires it; count failed conversions.
- [ ] Convert age, rating, coordinates, target, dates, and times to appropriate types where those fields exist.
- [ ] Count exact duplicates and repeated order IDs separately. Remove confirmed duplicates and explain ambiguous repeated IDs.
- [ ] Validate latitude in `[-90, 90]` and longitude in `[-180, 180]`. Inspect zero coordinates and implausible routes; zero is not automatically invalid everywhere.
- [ ] Investigate nonpositive delivery times, impossible ages/ratings, and extreme values. Use documented domain rules; do not remove rare valid delays just to improve scores.
- [ ] Remove rows without a usable target; never impute the target.
- [ ] Record remaining missing predictor values and plan median imputation for numeric fields and mode/constant imputation for categorical fields inside the training pipeline.
- [ ] Write cleaning functions in `src/data.py` and show before/after counts in the notebook.
- [ ] Maintain `reports/cleaning_log.md` with: issue, evidence, decision, affected rows, and reason.

**Done when:** cleaning can be rerun from the unchanged CSV and every deletion or transformation is explained.

### ETA-04 · Create and protect the 80/20 split — 30 min

- [ ] Separate predictors `X` and target `y`; exclude the target and outcome-derived fields from `X`.
- [ ] Use a fixed seed, for example `42`, for a reproducible 80/20 split. Save enough information to reproduce the exact split.
- [ ] If time ordering or related deliveries requires a chronological/group-aware split, document and implement that choice while keeping the required ratio as close as feasible.
- [ ] Reserve the test set for Thursday’s final evaluation. Perform detailed target-based EDA, feature selection, and model selection on the training portion.
- [ ] Keep imputation, scaling, category learning, and any learned outlier thresholds inside training/CV pipelines. Fixed parsing and documented validity checks can happen before splitting.

**Done when:** all models will receive the same split, and no preprocessing statistics have been learned from the test set.

**Daily buffer — 60 min:** fix parsing issues, check row counts, update Jira, and commit the data preparation milestone.

## 5. Tuesday — Explain the data and build features

### ETA-05 · Complete descriptive EDA — 120 min

- [ ] Calculate count, mean, median, standard deviation, and relevant quantiles for numeric variables; frequencies and proportions for categorical variables.
- [ ] Plot the target distribution and report its median, spread, and long tail.
- [ ] Analyze the target against **every usable predictor**: scatter/binned plots for numeric variables and boxplots or grouped summaries for categories. Explain excluded identifiers.
- [ ] Build and interpret a numeric correlation matrix. Do not encode unordered categories as arbitrary numbers just to correlate them.
- [ ] Look at distance, traffic, weather, courier rating/age, vehicle, and order timing where available. Show group counts to avoid overinterpreting tiny groups.
- [ ] Add one or two explanatory sentences under each useful chart and summarize the strongest associations. Association alone does not establish causation.

**Done when:** the notebook has a coherent explanation of which recorded factors relate to delivery duration.

### ETA-06 · Create at least two features — 75 min

- [ ] **Feature 1: `distance_km`.** Use the Haversine formula on restaurant/customer coordinates. Explain that straight-line distance approximates travel distance and does not capture the actual road route.
- [ ] **Feature 2: `order_hour`.** Extract the order hour from the order timestamp, if available. If it is absent, use a supported alternative such as day of week from an order date; document the actual inputs.
- [ ] Optionally derive a rush-hour flag or weekend flag only when the source data supports it and time remains.
- [ ] Validate feature calculations on a few manually checked records, including identical coordinates and a missing input.
- [ ] Put the calculations in `src/features.py`; reuse them through an importable pipeline transformer or a single shared preparation function at training and prediction time.
- [ ] Exclude actual pickup/completion times when they would not be known at the promised prediction moment. In particular, order-to-pickup delay is not available at order placement.

**Done when:** at least two new features are reproducible, explained, and computable for a new order without its outcome.

### ETA-07 · Implement preprocessing and a baseline — 90 min

- [ ] Build a `ColumnTransformer`: numeric imputation plus `StandardScaler` or `MinMaxScaler`; categorical imputation plus `OneHotEncoder(handle_unknown="ignore")`.
- [ ] Combine preprocessing and estimator in a scikit-learn `Pipeline` so every CV fold fits its own transformations.
- [ ] Establish a `DummyRegressor(strategy="median")` baseline. This is a reference point in addition to the four regression models planned for Wednesday.
- [ ] Use consistent cross-validation on the training portion, such as 5 shuffled folds with seed `42`; adapt the folds if you chose time/group-aware splitting.
- [ ] Record baseline CV MAE and RMSE. Convert negative scorer outputs back to positive errors before presenting them.

**Done when:** a raw input row passes through preparation and produces a baseline prediction without fitting anything on test data.

### ETA-08 · Consolidate the notebook — 30 min

- [ ] Organize sections: business objective, dataset, cleaning, split, EDA, features, preprocessing, baseline, models, optimization, evaluation, conclusions.
- [ ] Remove unused experiments and check that all plots have titles, units, and readable labels.
- [ ] Write a short list of dataset limitations and decisions to revisit in the final report.

**Daily buffer — 45 min:** rerun completed notebook sections, update Jira, and commit the EDA/feature milestone.

## 6. Wednesday — Compare and optimize models

### ETA-09 · Train four regression models — 120 min

- [ ] Fit **Linear Regression**, **Ridge Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** using the same training rows, eligible features, and CV folds.
- [ ] Keep preprocessing inside each pipeline. Apply the required numeric scaling; explain that it is useful for linear/regularized models while trees generally do not require it.
- [ ] Use fixed seeds for stochastic models and reasonable initial model sizes so experiments finish promptly.
- [ ] Collect CV MAE, MSE, RMSE, and R², including variability across folds, plus training time.
- [ ] Export the comparison to `reports/model_comparison.csv`. Add training scores where useful for diagnosing overfitting.

**Done when:** four models and the dummy baseline have a comparable, reproducible validation table.

### ETA-10 · Choose a candidate using validation — 45 min

- [ ] Use CV MAE as the primary selection metric because it expresses typical absolute error in minutes. Use RMSE to highlight large errors.
- [ ] Compare the candidate against the baseline and assess CV stability, overfitting, and runtime.
- [ ] Explain the choice in a paragraph. Prefer the simpler model when performance differences are negligible.
- [ ] Keep the test set untouched during this decision.

**Done when:** the chosen model and selection rule are documented before any test results are inspected.

### ETA-11 · Tune the candidate — 105 min

- [ ] Use `RandomizedSearchCV` or `GridSearchCV` on the **full pipeline**, fitting only on the training portion.
- [ ] Start with a bounded search, for example 10–15 candidates with 3 CV folds. Monitor runtime early and reduce the search if it threatens the schedule.
- [ ] Tune relevant parameters: for Ridge, `alpha`; for trees, depth and minimum leaf size; for Random Forest, also number of trees and maximum features. Use pipeline parameter names such as `model__max_depth`.
- [ ] Optimize MAE and record the search space, seed, best parameters, score, and search duration.
- [ ] Compare tuned and untuned candidates under the same final training-only CV scheme before choosing the final configuration.

**Done when:** the final configuration is fixed and its validation results justify the choice.

### ETA-12 · Prepare evaluation outputs — 30 min

- [ ] Implement metric calculation and actual-versus-predicted/residual plots in `src/evaluate.py`.
- [ ] Define useful error slices such as traffic category, weather, and distance bands, with minimum group counts.
- [ ] Prepare the final results table below without inventing values.

**Daily buffer — 60 min:** resolve model compatibility/runtime issues, update Jira, and commit the modeling milestone.

## 7. Thursday — Evaluate, save, and build the application

### ETA-13 · Perform the final held-out evaluation — 75 min

- [ ] Refit the selected configuration on the training portion and evaluate on the reserved test set.
- [ ] For the assignment’s comparison, evaluate all four fixed model configurations on the **same test rows**, reporting MAE, MSE, RMSE, and R². Keep the selected model based on validation; do not choose a new winner or retune after seeing test scores.
- [ ] Compare final-model train and test metrics, actual versus predicted values, and residuals. Describe signs of overfitting without assuming an ensemble automatically generalizes well.
- [ ] Report MAE and RMSE in minutes, MSE in minutes², and R² without units.
- [ ] Calculate adjusted R² when meaningful: `1 - (1 - R²) * (n - 1) / (n - p - 1)`, where `n` is the evaluation row count and `p` is the number of transformed predictor columns, excluding an intercept. If `n <= p + 1`, report it as undefined.
- [ ] Explain that adjusted R² is a classical linear-model statistic and is only a descriptive value for tree ensembles; it is not the tuning criterion.
- [ ] Inspect error slices and use permutation importance on validation data if time permits. Discuss limitations, rather than changing the model based on held-out errors.
- [ ] Write: “On the held-out test set, the model’s prediction differs from the observed time by **[MAE] minutes on average**.” Do not describe this as a guaranteed error bound.
- [ ] Describe observed conditions associated with long deliveries. A definitive “late” classification requires a promised delivery time or a separately justified business threshold.

**Done when:** all required metrics, plots, and a plain-language interpretation are saved and explained.

### ETA-14 · Save and reload the trained artifacts — 45 min

- [ ] Save the fitted pipeline to `models/delivery_eta_pipeline.joblib`.
- [ ] Export the **already fitted scaler** from the pipeline’s numeric preprocessing branch to `models/scaler.joblib` to satisfy the explicit deliverable. Do not fit a second scaler.
- [ ] Record input fields, feature definitions, package versions, seed, selected parameters, and model training scope in `models/metadata.json`.
- [ ] Implement `src/predict.py` so the app sends raw input fields through the saved pipeline and shared feature preparation. Avoid scaling inputs twice.
- [ ] Reload the artifact in a fresh process and verify predictions match the in-memory pipeline for a few rows.
- [ ] Keep the evaluated, training-split-fitted model as the delivered artifact for simplicity and traceability.

**Done when:** inference works without rerunning the notebook or retraining the model.

### ETA-15 · Build the Streamlit application — 150 min

- [ ] Add a prediction form containing the inputs actually required by the model: coordinates or distance, traffic, weather, vehicle type, courier rating, and order time as applicable.
- [ ] Keep input choices consistent with training categories. Explain units and validate coordinates, ratings, required fields, and time values.
- [ ] If training calculates distance from coordinates, collect coordinates and reuse that calculation. Only offer direct-distance entry if the inference preparation explicitly supports it.
- [ ] Load the saved pipeline once using Streamlit resource caching.
- [ ] Display **“Estimated delivery time: X minutes”**, rounded sensibly, with a clear submit button and useful validation messages.
- [ ] Show a **Data** section with target distribution and traffic/weather/distance charts from training EDA.
- [ ] Show a **Model performance** section with held-out MAE, RMSE, R², actual-versus-predicted plot, and residual/error distribution. For regression, avoid an unexplained “accuracy percentage.”
- [ ] Explain that this is an estimate from historical data. Do not display an arbitrary ± range as a statistically validated prediction interval.
- [ ] Keep the interface usable by a nontechnical person and ensure all displayed metrics come from saved evaluation outputs.

**Done when:** a user can enter a delivery, receive an ETA, and understand the model’s measured error without opening the notebook.

### ETA-16 · Check the full prediction flow — 30 min

- [ ] Try at least three valid scenarios, including ordinary traffic and a more difficult condition represented in the dataset.
- [ ] Try missing/invalid inputs and confirm that errors are understandable.
- [ ] Confirm that reloading the app does not retrain the model and that inference agrees with the shared prediction code.

**Daily buffer — 60 min:** fix application issues, update Jira, and commit the working end-to-end solution.

## 8. Friday — Verify, document, and submit

### ETA-17 · Verify reproducibility — 90 min

- [ ] Restart the notebook kernel and run every cell in order. Fix hidden state, missing files, and stale outputs.
- [ ] Rebuild from a clean Docker cache with the finalized `requirements.txt` and follow the README commands exactly.
- [ ] Confirm training/evaluation commands run and the saved model reloads correctly.
- [ ] Launch Streamlit from the repository root and exercise the main form and performance section.
- [ ] Check that paths are relative to the project or resolved from its location, rather than tied to your machine.
- [ ] Record the verified Python version and package versions.

**Done when:** someone with the documented dataset can reproduce the workflow and start the application.

### ETA-18 · Finalize the README and Jira — 75 min

- [ ] Fill in dataset source, size, target definition, cleaning decisions, chosen features, split strategy, selected model, metrics, and limitations.
- [ ] Replace proposed commands with the commands that actually worked.
- [ ] Capture two screenshots: the prediction interface with a result and the model-performance view. Save them under `reports/screenshots/` and embed them in this README.
- [ ] Complete Jira statuses and add evidence/links for completed issues. Move unfinished bonuses out of required delivery scope.
- [ ] Verify repository and Jira access for the evaluator. If raw data cannot be redistributed, document exactly how to obtain it and where to place it.

**Done when:** the repository explains what was built, how well it works, and how to run it.

### ETA-19 · Rehearse the assessment — 60 min

Use this 10-minute demonstration outline:

| Time | Show and explain |
| --- | --- |
| 0:00–1:00 | Business question, intended prediction moment, target definition |
| 1:00–3:00 | Main cleaning decisions and two useful EDA findings |
| 3:00–4:30 | Engineered features and how the pipeline prevents leakage |
| 4:30–6:30 | Four-model comparison, tuning, final errors, overfitting check |
| 6:30–9:00 | Live Streamlit prediction and performance visualizations |
| 9:00–10:00 | Limitations and next steps |

- [ ] Practice answering: Why MAE? Why this split? Why this imputation/scaling? Why this model? What is leakage? How would you handle an unseen category? What causes large errors?
- [ ] Be ready to navigate `src/`, explain a pipeline step, change an input or model parameter, and trace a prediction during the code review/practical exercise.
- [ ] Keep screenshots and saved notebook outputs available if the live demonstration has a problem.

### ETA-20 · Submit and preserve a final version — 45 min

- [ ] Recheck every required deliverable in Section 1.
- [ ] Verify no secrets, temporary files, or broken links are included.
- [ ] Commit and push the final work; verify the remote repository includes the notebook, app, requirements, and trained artifacts.
- [ ] Submit the repository link and Jira link through the required channel before the official deadline. Confirm any additional submission instructions with the course platform.
- [ ] Record the submitted commit and keep a local copy.

**Final buffer — 90 min:** reserve for installation, notebook, artifact, or submission problems. Aim to finish at 18:00; do not plan to use the midnight cutoff as ordinary working time.

## 9. Rules that protect the schedule

1. **Finish the required workflow first.** Cleaning → EDA → features → pipeline → four models → tuning → evaluation → Streamlit → delivery.
2. **Timebox investigation.** If a problem consumes 30 minutes without progress, write down the error and try the simplest valid solution within the brief.
3. **Keep experiments bounded.** Smaller tuning searches are better than unfinished deliverables. Do not add extra model families before the required comparison works.
4. **Keep the app simple.** A clear form, prediction, charts, and measured errors are sufficient. Defer elaborate styling and deployment unless required elsewhere.
5. **If Monday slips:** use the next buffer for cleaning; simplify EDA presentation without skipping required variable analysis.
6. **If Wednesday slips:** narrow the search and finish all four models; remove bonuses immediately.
7. **If Thursday slips:** spend Friday’s buffer completing and checking the app; preserve documentation, submission, and a short rehearsal.
8. **Only start a bonus after the core checklist and reproducibility checks pass**, with at least two spare hours outside the final buffer. Choose one, keep it isolated, and stop if it threatens delivery.

## 10. Docker installation and execution — finalize after implementation

These are the **intended commands** for the planned structure. The current Docker and source files still need implementation; test and adjust these instructions before submission.

Prerequisites: Docker Engine with the Compose plugin. From the repository root:

```bash
docker compose build
```

The original dataset is currently stored at `data/raw/dataset-1.csv`. Keep that path consistent in the code and final instructions. The same Compose service should be reused for notebook, training, evaluation, and application commands so every stage uses the same dependency versions.

Planned commands:

```bash
# Open the complete analysis notebook
docker compose run --rm --service-ports app \
  jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root notebooks/01_delivery_eta.ipynb

# Train, compare, tune, and save the selected pipeline
docker compose run --rm app python -m src.train

# Evaluate saved configurations on the held-out split and export reports
docker compose run --rm app python -m src.evaluate

# Run the user application
docker compose run --rm --service-ports app \
  python -m streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

The service name `app` is a proposed convention; update these commands if `compose.yaml` uses another name. The application should load the provided trained artifact and run without needing to train first. Document the actual ports, command arguments, volume behavior, and runtime expectations here once verified.

## 11. Final project record — fill in before delivery

### Dataset and approach

- **Dataset source/download instructions:** Pending.
- **Raw / cleaned rows and columns:** Pending.
- **Target definition and unit:** Pending verification; intended output is minutes.
- **Inputs available at prediction time:** Pending.
- **Main cleaning decisions and removed-row counts:** Pending; see `reports/cleaning_log.md` when created.
- **New features and their rationale:** Pending.
- **Split and cross-validation strategy:** Pending.
- **Selected model and best parameters:** Pending.
- **Verified Python version:** Pending.
- **Known limitations:** Pending findings about target definition, geographic coverage, missing inputs, straight-line distance, and difficult delivery conditions.

### Actual results

Do not enter estimates or invented scores. Add CV results separately from the final held-out results and label both clearly.

| Model | CV MAE (min) | Test MAE (min) | Test MSE (min²) | Test RMSE (min) | Test R² |
| --- | --- | --- | --- | --- | --- |
| Dummy median baseline | Pending | Pending | Pending | Pending | Pending |
| Linear Regression | Pending | Pending | Pending | Pending | Pending |
| Ridge Regression | Pending | Pending | Pending | Pending | Pending |
| Decision Tree | Pending | Pending | Pending | Pending | Pending |
| Random Forest | Pending | Pending | Pending | Pending | Pending |
| Tuned selected model | Pending | Pending | Pending | Pending | Pending |

- **Final model training versus test metrics:** Pending.
- **Final model adjusted R², `n`, and `p`:** Pending; state if undefined and explain its limitations.
- **Business interpretation:** Pending measured MAE and explanation of the most difficult conditions.
- **Application screenshots:** Pending; embed the actual files after capture.
- **Repository link:** Pending.
- **Jira board link:** Pending.
- **Submitted commit and submission time:** Pending.

You are ready to submit when the required checklist is complete, the app runs from the saved model, the workflow can be reproduced, and you can explain your decisions within the assessment time.
