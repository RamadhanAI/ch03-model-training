# Chapter 3: Model Training
This chapter explores supervised and unsupervised learning using credit card fraud detection and customer segmentation.
# Chapter 3: Model Training — Fraud Detection & Customer Clustering

This chapter demonstrates how to build real-world supervised and unsupervised models with a production mindset. We train fraud classifiers, explain models with SHAP, log experiments using MLflow, and run the pipeline from both a notebook and the CLI.

---

## 🧠 Key Features

- Logistic Regression & Random Forest classification
- Custom threshold tuning for imbalanced data
- SHAP explainability visualizations
- KMeans clustering with silhouette scoring
- MLflow model tracking (with local UI)
- CI-ready GitHub Actions pipeline
- CLI script for end-to-end training
- Visual performance reporting

---

## 📁 Folder Structure

```bash
ch03-model-training/
├── data/                    # Place creditcard.csv here
├── notebooks/               # Interactive notebook (training, SHAP, clustering)
├── src/                     # Python modules & CLI training script
├── models/                  # Saved model files (pkl, onnx)
├── plots/                   # Precision-recall curve output
├── tests/                   # Latency check script
├── .github/workflows/       # CI with GitHub Actions
├── README.md
└── requirements.txt
📥 Dataset Download

The dataset is not included in this repo due to GitHub's file size limits.

👉 Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file as:

data/creditcard.csv
🚀 Run the Training CLI

To run the full pipeline (preprocessing → training → logging):

python src/train_classifier.py
Outputs:

models/fraud_model.pkl
plots/precision_recall_curve.png
MLflow logs (see below)
🔍 MLflow Tracking (Optional)

To view your experiments locally:

mlflow ui
Then open http://localhost:5000 in your browser.

📊 Visual Output

Precision-Recall Curve from train_classifier.py:
plots/
└── precision_recall_curve.png
🔁 GitHub CI

Automated validation with GitHub Actions runs on .py and .ipynb changes.

Check .github/workflows/validate.yml for CI config.

✨ Contributing

Feel free to fork, clone, or adapt for your own MLOps capstone project.
Built as part of Applied AI and MLOps: From Idea to Deployment.

🔗 License

MIT License
