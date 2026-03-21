# tourism-popularity-prediction
Tourism popularity prediction using Google Trends and Reddit sentiment


## Current Results

Baseline model trained using Google Trends features:

Features:
- trend_score
- city_code
- month

Target:
- future popularity class

Model:
RandomForestClassifier

Accuracy:
~0.69
“Initial baseline experiment”
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Baseline Model (Google Trends Only)

A baseline model was built using only Google Trends data.

Features:
- trend (monthly)
- city_code
- month

Target:
- future popularity class (t+1)

Model:
- RandomForestClassifier

Results:
- Accuracy: ~0.72
- Balanced performance across classes

This serves as the reference model to evaluate the contribution of Reddit sentiment data.
