# DeliveryETA — Project Structure

```text
DeliveryETA/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   ├── dataset-1.csv
│   │   └── dataset-2.csv
│   └── processed/
├── models/
│   ├── delivery_eta_pipeline.joblib
│   └── scaler.joblib
├── notebooks/
│   └── 01_delivery_eta.ipynb
├── reports/
│   ├── figures/
│   ├── screenshots/
│   ├── cleaning_log.md
│   ├── final_metrics.json
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   └── training_eda.csv
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── model_config.py
│   ├── train.py
│   ├── tuning.py
│   └── validation.py
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile
├── README.md
└── requirements.txt
```
