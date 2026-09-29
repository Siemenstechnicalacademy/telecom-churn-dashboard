import joblib
import pandas as pd
import numpy as np

def predict_literature_paper_performance(paper_dict, model_name='best'):
    """
    Loads models trained on the 30 Research Papers dataset and predicts
    whether a given research methodology configuration achieves High Performance (Acc >= 85%).
    """
    model_bundle = joblib.load("churn_model.joblib")
    models = model_bundle.get('models', {})
    scaler = model_bundle.get('scaler', None)
    feature_names = model_bundle['feature_names']
    best_model_name = model_bundle.get('best_model_name', 'Logistic Regression')

    df = pd.DataFrame([paper_dict])

    # Feature engineering for research paper methodology
    if 'advanced_tech_score' not in df.columns:
        df['advanced_tech_score'] = (
            df.get('has_xgboost', 0) +
            df.get('has_deep_learning', 0) +
            df.get('has_imbalance_handling', 0) +
            df.get('has_feature_selection', 0)
        )
    if 'modern_methodology_index' not in df.columns:
        df['modern_methodology_index'] = (df.get('year', 2024) - 2020) * df.get('is_empirical_study', 1)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]

    results = {}
    for name, m in models.items():
        if name == 'Logistic Regression' and scaler is not None:
            input_data = scaler.transform(df)
        else:
            input_data = df

        prob = m.predict_proba(input_data)[0][1]
        pred = m.predict(input_data)[0]
        results[name] = {
            'prediction': int(pred),
            'high_performance_prob': float(prob),
            'probability_percent': round(float(prob) * 100, 2)
        }

    print("\n" + "=" * 65)
    print(" 30 RESEARCH PAPERS LITERATURE DATASET: PREDICTION COMPARISON")
    print("=" * 65)
    for name, res in results.items():
        label = "[HIGH ACCURACY RESEARCH METHODOLOGY (>=85%)]" if res['prediction'] == 1 else "[BASELINE METHODOLOGY (<85%)]"
        is_best = " [BEST MODEL]" if name == best_model_name else ""
        print(f" * {name:<20}{is_best}: {res['probability_percent']:>6.2f}% Prob  -->  {label}")
    print("=" * 65)

    if model_name.lower() == 'best' or model_name not in results:
        selected_res = results[best_model_name]
    else:
        selected_res = results[model_name]

    return selected_res['prediction'], selected_res['high_performance_prob'], results

if __name__ == "__main__":
    sample_paper_config = {
        'year': 2025,
        'has_logistic_regression': 1,
        'has_random_forest': 1,
        'has_xgboost': 1,
        'has_deep_learning': 0,
        'has_imbalance_handling': 1,
        'has_explainability': 1,
        'has_feature_selection': 1,
        'is_empirical_study': 1,
        'is_ibm_telco_dataset': 1
    }

    predict_literature_paper_performance(sample_paper_config)
