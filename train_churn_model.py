import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def load_and_extract_literature_data(excel_path=r"Telecom_Customer_Churn_Literature_Analysis (1).xlsx"):
    """
    Loads research paper data directly from Telecom_Customer_Churn_Literature_Analysis (1).xlsx,
    parsing textual research paper entries into structured tabular features.
    """
    if not os.path.exists(excel_path):
        alt_path = r"C:\Users\ADMIN\Downloads\Telecom_Customer_Churn_Literature_Analysis (1).xlsx"
        if os.path.exists(alt_path):
            excel_path = alt_path
        else:
            raise FileNotFoundError(f"Literature analysis Excel file not found at {excel_path}")

    print(f"--> [STEP 2] Loading research paper data from Excel: {os.path.abspath(excel_path)}")
    df_raw = pd.read_excel(excel_path, sheet_name='Literature Analysis')

    records = []
    for idx, row in df_raw.iterrows():
        text_tech = str(row['Technique Used']).lower()
        text_acc = str(row['Accuracy / Main Result']).lower()
        text_ds = str(row['Dataset Used']).lower()
        text_type = str(row['Paper Type']).lower()

        acc_matches = re.findall(r'(\d{2}(?:\.\d+)?)\s*%', text_acc)
        acc_vals = [float(x) for x in acc_matches if 50.0 <= float(x) <= 100.0]
        max_acc = max(acc_vals) if acc_vals else (88.0 if 'accuracy' in text_acc or 'verified' in str(row['Verification Status']).lower() else 78.0)

        records.append({
            'paper_id': int(row['Sr.No']),
            'year': int(row['Year']),
            'has_logistic_regression': 1 if 'logistic' in text_tech or 'lr' in text_tech else 0,
            'has_random_forest': 1 if 'random forest' in text_tech or 'rf' in text_tech else 0,
            'has_xgboost': 1 if 'xgboost' in text_tech or 'boost' in text_tech else 0,
            'has_deep_learning': 1 if any(k in text_tech for k in ['cnn', 'lstm', 'neural', 'ann', 'bilstm', 'deep']) else 0,
            'has_imbalance_handling': 1 if any(k in text_tech + text_acc for k in ['smote', 'adasyn', 'imbalance', 'class']) else 0,
            'has_explainability': 1 if any(k in text_tech + text_acc for k in ['shap', 'lime', 'explain']) else 0,
            'has_feature_selection': 1 if any(k in text_tech + text_acc for k in ['feature selection', 'lasso', 'optimization', 'aoa']) else 0,
            'is_empirical_study': 1 if 'empirical' in text_type or 'original' in text_type or 'journal' in text_type else 0,
            'is_ibm_telco_dataset': 1 if any(k in text_ds for k in ['ibm', 'kaggle', 'telco', 'uci']) else 0,
            'reported_max_accuracy': max_acc,
            'high_accuracy_target': 1 if max_acc >= 85.0 else 0
        })

    df = pd.DataFrame(records)
    
    df['advanced_tech_score'] = df['has_xgboost'] + df['has_deep_learning'] + df['has_imbalance_handling'] + df['has_feature_selection']
    df['modern_methodology_index'] = (df['year'] - 2020) * df['is_empirical_study']

    X = df.drop(columns=['paper_id', 'reported_max_accuracy', 'high_accuracy_target'])
    y = df['high_accuracy_target']

    return X, y, list(X.columns), df

def train_and_evaluate_literature_models(excel_path=r"Telecom_Customer_Churn_Literature_Analysis (1).xlsx"):
    """
    Trains Logistic Regression, Random Forest, and Gradient Boosting (XGBoost alternative) on Excel dataset.
    """
    print("\n" + "=" * 75)
    print(" EXCEL RESEARCH PAPERS DATASET: MULTI-MODEL ML PIPELINE")
    print("=" * 75)

    X, y, feature_names, df_full = load_and_extract_literature_data(excel_path)

    print("\n[STEP 2] 30 RESEARCH PAPERS DATASET SUMMARY")
    print(f" - Total Research Papers : {len(df_full)}")
    print(f" - Extracted Features    : {X.shape[1]}")
    print(f" - High Performance (1) : {(y == 1).sum()} papers (Acc >= 85%)")
    print(f" - Baseline Performance (0): {(y == 0).sum()} papers (Acc < 85%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        'XGBoost': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = {}
    print("\n[STEP 6] EXECUTING CROSS-VALIDATION ON LITERATURE DATASET...")

    for name, model in models.items():
        if name == 'Logistic Regression':
            cv_res = cross_validate(model, X_train_scaled, y_train_res, cv=skf, scoring=['accuracy', 'f1', 'roc_auc'])
        else:
            cv_res = cross_validate(model, X_train_res, y_train_res, cv=skf, scoring=['accuracy', 'f1', 'roc_auc'])

        cv_acc = cv_res['test_accuracy'].mean()
        cv_f1 = cv_res['test_f1'].mean()
        cv_auc = cv_res['test_roc_auc'].mean()
        cv_scores[name] = {'CV Accuracy': cv_acc, 'CV F1': cv_f1, 'CV ROC-AUC': cv_auc}
        print(f" - {name:<20} | 3-Fold CV Acc: {cv_acc:.4f} | CV F1: {cv_f1:.4f} | CV ROC-AUC: {cv_auc:.4f}")

    results = {}
    fitted_models = {}
    roc_curves = {}

    print("\n" + "=" * 75)
    print(" [STEP 7] TEST SET EVALUATION (HELD-OUT RESEARCH PAPERS)")
    print("=" * 75)

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train_res)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        fitted_models[name] = model
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'CV Accuracy': cv_scores[name]['CV Accuracy'],
            'Confusion Matrix': cm.tolist()
        }

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)

    comparison_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Accuracy']]
    print(comparison_df.round(4).to_string())

    best_model_name = comparison_df['Accuracy'].idxmax()
    print(f"\n[SUMMARY] Best Performing Model on 30 Research Papers Dataset: {best_model_name}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    model_names = list(models.keys())
    for idx, name in enumerate(model_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        cm = np.array(results[name]['Confusion Matrix'])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Baseline Acc", "High Acc (>=85%)"],
                    yticklabels=["Baseline Acc", "High Acc (>=85%)"])
        ax.set_title(f"{name} Confusion Matrix", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

    ax_roc = axes[1, 1]
    for name, (fpr, tpr, auc) in roc_curves.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax_roc.set_title("ROC Curve Comparison (Research Papers)", fontsize=11, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    best_cm = np.array(results[best_model_name]['Confusion Matrix'])
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Baseline Acc", "High Acc"],
                yticklabels=["Baseline Acc", "High Acc"])
    plt.title(f"Confusion Matrix - {best_model_name} (Best Literature Model)", fontsize=11)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    model_bundle = {
        'model': fitted_models[best_model_name],
        'best_model_name': best_model_name,
        'models': fitted_models,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': results,
        'comparison_table': comparison_df.to_dict()
    }

    joblib.dump(model_bundle, "churn_model.joblib")
    joblib.dump(fitted_models[best_model_name], "churn_model.pkl")

    print("\n[SUCCESS] Trained Pure Scikit-Learn Model Bundle Saved to churn_model.joblib!")
    return model_bundle

if __name__ == "__main__":
    train_and_evaluate_literature_models()
