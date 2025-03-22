# Diagnosing Sleep Apnea Using Thermal Imagery

## Problem Statement:
Model respiration signal time-series data to estimate apnea events using sequence modeling techniques.

## Approach:
1. **Data Preprocessing**:
   - Handle missing values using linear interpolation.
   - Normalize data using MinMaxScaler.
   - Segment time-series data into overlapping windows.

2. **Feature Extraction**:
   - Extract statistical, temporal, and spectral features using TSFEL.

3. **Modeling**:
   - Train an LSTM-based neural network to classify apnea events.

## Requirements:
- Python 3.x
- Libraries: TensorFlow, TSFEL, pandas, scikit-learn

## How to Run:
1. Clone this repository:
    - git clone https://github.com/Harsimran-Dalal/sleep-apnea-model.git
    - cd sleep-apnea-model

2. Install dependencies:
    - pip install -r requirements.txt

3. Preprocess the data:
    - python preprocessing.py

4. Extract features:
    - python feature_extraction.py

5. Train and evaluate the model:
    - python model.py

## Results:
- Model achieved ~90% accuracy on test data.
- F1-score: ~0.88 for apnea detection.

## Future Improvements:
- Incorporate additional signals like SpO2 or airflow.
- Explore hybrid models like CNN-LSTMs or Transformers.
