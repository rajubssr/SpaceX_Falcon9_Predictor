# SpaceX Falcon 9 First-Stage Landing Success Predictor

Predicts whether a Falcon 9 first-stage booster will successfully land, using 4 classification algorithms benchmarked against each other.

## Models Benchmarked
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)

## Tech Stack
Python · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## Pipeline
1. Data collection via SpaceX public API
2. EDA and visualization
3. Feature engineering + preprocessing
4. Model training & hyperparameter tuning
5. Cross-validated evaluation (Accuracy, Precision, Recall, F1)

## Setup & Run

```bash
pip install -r requirements.txt
python data_collection.py   # Fetch data from SpaceX API
python eda.py               # Exploratory data analysis
python train.py             # Train & benchmark all models
```

## Results
| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | ~83%     | ~0.82    |
| SVM                 | ~85%     | ~0.84    |
| Decision Tree       | ~80%     | ~0.79    |
| KNN                 | ~82%     | ~0.81    |
