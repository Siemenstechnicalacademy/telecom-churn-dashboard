import joblib
import pandas as pd

def run_realtime_cli():
    print("=" * 65)
    print(" 30 RESEARCH PAPERS EXCEL DATA MODEL SUITE (CLI INTERACTIVE)")
    print("=" * 65)

    model_bundle = joblib.load("churn_model.joblib")
    models = model_bundle.get('models', {})
    scaler = model_bundle.get('scaler', None)
    feature_names = model_bundle['feature_names']
    best_model_name = model_bundle.get('best_model_name', 'Logistic Regression')

    while True:
        print("\nEnter Research Paper Configuration details (or type 'exit' to quit):")
        try:
            inp = input("Publication Year (default 2025): ").strip()
            if inp.lower() == 'exit': break
            year = float(inp) if inp else 2025.0

            inp = input("Evaluates Logistic Regression [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_lr = float(inp) if inp else 1.0

            inp = input("Evaluates Random Forest [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_rf = float(inp) if inp else 1.0

            inp = input("Evaluates XGBoost [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_xgb = float(inp) if inp else 1.0

            inp = input("Evaluates Deep Learning CNN/LSTM [1/0] (default 0): ").strip()
            if inp.lower() == 'exit': break
            has_dl = float(inp) if inp else 0.0

            inp = input("Uses SMOTE Imbalance Handling [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_smote = float(inp) if inp else 1.0

            inp = input("Uses SHAP/LIME Explainability [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_exp = float(inp) if inp else 1.0

            inp = input("Uses Feature Selection [1/0] (default 1): ").strip()
            if inp.lower() == 'exit': break
            has_fs = float(inp) if inp else 1.0

            data = {
                'year': year,
                'has_logistic_regression': has_lr,
                'has_random_forest': has_rf,
                'has_xgboost': has_xgb,
                'has_deep_learning': has_dl,
                'has_imbalance_handling': has_smote,
                'has_explainability': has_exp,
                'has_feature_selection': has_fs,
                'is_empirical_study': 1.0,
                'is_ibm_telco_dataset': 1.0,
                'advanced_tech_score': has_xgb + has_dl + has_smote + has_fs,
                'modern_methodology_index': (year - 2020) * 1.0
            }

            df = pd.DataFrame([data])
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]

            print("\n" + "-" * 60)
            print(" PREDICTION RESULTS ACROSS MODELS (30 RESEARCH PAPERS EXCEL DATA):")
            print("-" * 60)

            for name, m in models.items():
                if name == 'Logistic Regression' and scaler is not None:
                    inp_df = scaler.transform(df)
                else:
                    inp_df = df

                prob = m.predict_proba(inp_df)[0][1] * 100
                pred = m.predict(inp_df)[0]
                status_tag = "HIGH ACCURACY METHODOLOGY (>=85%)" if pred == 1 else "BASELINE METHODOLOGY (<85%)"
                star = " [BEST MODEL]" if name == best_model_name else ""
                print(f" * {name:<20}{star}: {prob:>6.2f}% Prob | {status_tag}")

            print("-" * 60)

        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")

if __name__ == "__main__":
    run_realtime_cli()
