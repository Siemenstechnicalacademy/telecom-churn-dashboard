import os
import re
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load literature dataset and trained model bundle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "Telecom_Customer_Churn_Literature_Analysis (1).xlsx")
if not os.path.exists(EXCEL_PATH):
    EXCEL_PATH = r"C:\Users\ADMIN\Downloads\Telecom_Customer_Churn_Literature_Analysis (1).xlsx"

MODEL_PATH = os.path.join(BASE_DIR, "churn_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model churn_model.joblib not found! Please run train_churn_model.py first.")

model_bundle = joblib.load(MODEL_PATH)
models = model_bundle.get('models', {})
scaler = model_bundle.get('scaler', None)
feature_names = model_bundle['feature_names']
best_model_name = model_bundle.get('best_model_name', 'Logistic Regression')
metrics_data = model_bundle.get('metrics', {})

# Load 30 Research Papers raw dataset for dashboard exploration
def load_all_papers_json():
    if os.path.exists(EXCEL_PATH):
        df_raw = pd.read_excel(EXCEL_PATH, sheet_name='Literature Analysis')
        cols = ['Sr.No', 'Title of the Paper', 'Authors', 'Year', 'Technique Used', 'Accuracy / Main Result', 'Dataset Used', 'Verification Status', 'Source URL']
        df_sub = df_raw[cols].fillna('N/A')
        return df_sub.to_dict(orient='records')
    return []

all_papers = load_all_papers_json()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telecom Churn AI Dashboard | 30 Research Papers Literature Suite</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #070a12;
            --card-bg: rgba(18, 26, 43, 0.75);
            --card-hover: rgba(30, 41, 65, 0.85);
            --border: rgba(255, 255, 255, 0.12);
            --primary: #6366f1;
            --primary-light: #818cf8;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }

        body {
            background: radial-gradient(circle at 10% 10%, #17153b 0%, #070a12 60%, #030509 100%);
            color: var(--text); min-height: 100vh; padding: 2rem 1.5rem;
        }

        .dashboard-container {
            max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 2rem;
        }

        .navbar {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 20px; padding: 1.25rem 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .navbar .brand { display: flex; align-items: center; gap: 0.8rem; }
        .navbar .brand-icon {
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            display: flex; align-items: center; justify-content: center; font-size: 1.4rem; font-weight: 700;
        }

        .navbar h1 {
            font-size: 1.5rem; font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #6366f1, #38bdf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .navbar .tag {
            background: rgba(99, 102, 241, 0.2); border: 1px solid var(--primary);
            color: #a5b4fc; padding: 0.4rem 1rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;
        }

        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem;
        }

        .stat-card {
            background: var(--card-bg); backdrop-filter: blur(16px); border: 1px solid var(--border);
            border-radius: 20px; padding: 1.5rem; display: flex; flex-direction: column; gap: 0.5rem;
            transition: all 0.3s ease; position: relative; overflow: hidden;
        }

        .stat-card:hover { transform: translateY(-4px); border-color: var(--primary); box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25); }

        .stat-card .label { font-size: 0.88rem; color: var(--text-muted); font-weight: 500; }
        .stat-card .val { font-size: 2.2rem; font-weight: 700; color: #fff; }
        .stat-card .subtext { font-size: 0.8rem; color: var(--accent-emerald); }

        .main-section-grid {
            display: grid; grid-template-columns: 1fr 1.3fr; gap: 2rem;
        }

        @media (max-width: 1024px) { .main-section-grid { grid-template-columns: 1fr; } }

        .card-box {
            background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 24px; padding: 2rem; display: flex; flex-direction: column; gap: 1.5rem;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }

        .card-box h2 {
            font-size: 1.3rem; font-weight: 700; color: #cbd5e1;
            display: flex; align-items: center; justify-content: space-between;
        }

        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
        @media (max-width: 600px) { .form-grid { grid-template-columns: 1fr; } }

        .input-group { display: flex; flex-direction: column; gap: 0.4rem; }
        .input-group label { font-size: 0.85rem; font-weight: 500; color: #94a3b8; display: flex; justify-content: space-between; }
        .input-group input, .input-group select {
            background: rgba(10, 15, 28, 0.7); border: 1px solid var(--border); border-radius: 12px;
            padding: 0.75rem 1rem; color: #fff; font-size: 0.95rem; outline: none; transition: all 0.2s ease;
        }
        .input-group input:focus, .input-group select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25); }

        .btn-predict {
            grid-column: span 2; margin-top: 0.5rem; background: linear-gradient(135deg, #6366f1, #06b6d4);
            color: #fff; border: none; border-radius: 12px; padding: 1rem; font-size: 1.05rem; font-weight: 600;
            cursor: pointer; transition: all 0.3s ease; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
        }
        .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 15px 30px rgba(99, 102, 241, 0.6); }

        .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; }
        .res-card {
            background: rgba(10, 15, 28, 0.8); border: 1px solid var(--border); border-radius: 16px;
            padding: 1.25rem; text-align: center; transition: all 0.3s ease;
        }
        .res-card.active-best { border-color: var(--primary); box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
        .res-card .m-title { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; margin-bottom: 0.5rem; }
        .res-card .m-prob { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }

        .badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
        .badge-high { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
        .badge-low { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

        /* Research Papers Explorer Table & Search */
        .search-bar-box { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
        .search-input {
            flex: 1; background: rgba(10, 15, 28, 0.7); border: 1px solid var(--border); border-radius: 12px;
            padding: 0.75rem 1.2rem; color: #fff; font-size: 0.95rem; outline: none;
        }

        .papers-table-wrapper {
            max-height: 480px; overflow-y: auto; border: 1px solid var(--border); border-radius: 16px;
        }

        table { width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: left; }
        th, td { padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); }
        th { background: rgba(10, 15, 28, 0.9); color: var(--text-muted); font-weight: 600; sticky: top; }
        tr:hover { background: rgba(255, 255, 255, 0.03); }

        .link-doi { color: #38bdf8; text-decoration: none; font-weight: 500; }
        .link-doi:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Top Navigation -->
        <div class="navbar">
            <div class="brand">
                <div class="brand-icon">⚡</div>
                <div>
                    <h1>Telecom Churn AI Dashboard</h1>
                    <p style="font-size: 0.85rem; color: var(--text-muted);">Multi-Model Research Suite Trained Exclusively on Excel Dataset</p>
                </div>
            </div>
            <div class="tag">30 Research Papers Baseline</div>
        </div>

        <!-- KPI Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Literature Papers</div>
                <div class="val">30</div>
                <div class="subtext">Verified Research Studies</div>
            </div>

            <div class="stat-card">
                <div class="label">High Accuracy Models (≥85%)</div>
                <div class="val">25</div>
                <div class="subtext">83.3% Literature Coverage</div>
            </div>

            <div class="stat-card">
                <div class="label">Peak Literature Accuracy</div>
                <div class="val">97.80%</div>
                <div class="subtext">RF + XGBoost Hybrid Model</div>
            </div>

            <div class="stat-card">
                <div class="label">Best Excel Model (F1-Score)</div>
                <div class="val" style="color: #a5b4fc;">Logistic Reg</div>
                <div class="subtext">83.3% Test Acc | 0.889 F1-Score</div>
            </div>
        </div>

        <!-- Main Section Grid -->
        <div class="main-section-grid">
            <!-- Left Panel: Live ML Predictor -->
            <div class="card-box">
                <h2>⚡ Interactive Methodology Predictor</h2>
                <form id="methodologyForm" class="form-grid">
                    <div class="input-group">
                        <label>Publication Year <span><b id="yearVal">2025</b></span></label>
                        <input type="range" id="year" min="2020" max="2026" value="2025" oninput="document.getElementById('yearVal').innerText = this.value; fetchPrediction();">
                    </div>

                    <div class="input-group">
                        <label>Logistic Regression</label>
                        <select id="has_logistic_regression" onchange="fetchPrediction();">
                            <option value="1">Evaluated (Yes)</option>
                            <option value="0">Not Evaluated (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Random Forest</label>
                        <select id="has_random_forest" onchange="fetchPrediction();">
                            <option value="1">Evaluated (Yes)</option>
                            <option value="0">Not Evaluated (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>XGBoost / Boosting</label>
                        <select id="has_xgboost" onchange="fetchPrediction();">
                            <option value="1">Evaluated (Yes)</option>
                            <option value="0">Not Evaluated (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Deep Learning (CNN/LSTM)</label>
                        <select id="has_deep_learning" onchange="fetchPrediction();">
                            <option value="1">Evaluated (Yes)</option>
                            <option value="0">Not Evaluated (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>SMOTE Imbalance Handling</label>
                        <select id="has_imbalance_handling" onchange="fetchPrediction();">
                            <option value="1">Applied (Yes)</option>
                            <option value="0">Not Applied (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>SHAP / LIME Explainability</label>
                        <select id="has_explainability" onchange="fetchPrediction();">
                            <option value="1">Included (Yes)</option>
                            <option value="0">Not Included (No)</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Feature Selection Strategy</label>
                        <select id="has_feature_selection" onchange="fetchPrediction();">
                            <option value="1">Applied (Yes)</option>
                            <option value="0">Not Applied (No)</option>
                        </select>
                    </div>

                    <button type="submit" class="btn-predict">Predict Literature Performance Live 🚀</button>
                </form>

                <div class="results-grid" id="resultsGrid">
                    <!-- Live model risk gauges injected here -->
                </div>
            </div>

            <!-- Right Panel: Model Evaluation Benchmark -->
            <div class="card-box">
                <h2>📈 Model Benchmark on 30 Excel Papers</h2>
                <div class="papers-table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Model Name</th>
                                <th>Test Acc</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>3-Fold CV Acc</th>
                                <th>ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody id="benchmarkTableBody">
                            <!-- Injected dynamically -->
                        </tbody>
                    </table>
                </div>

                <div style="background: rgba(10, 15, 28, 0.6); border: 1px dashed var(--border); border-radius: 16px; padding: 1.25rem; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
                    <strong>💡 Key Literature Insight:</strong>
                    <p id="insightText">Loading literature research conclusions...</p>
                </div>
            </div>
        </div>

        <!-- Bottom Section: 30 Research Papers Explorer Table -->
        <div class="card-box">
            <h2>
                <span>📑 30 Telecom Churn Research Papers Explorer</span>
                <small style="font-size: 0.85rem; color: var(--text-muted); font-weight: normal;">Searchable Database from Telecom_Customer_Churn_Literature_Analysis (1).xlsx</small>
            </h2>

            <div class="search-bar-box">
                <input type="text" id="searchInput" class="search-input" placeholder="🔍 Search papers by title, authors, models (e.g. XGBoost, SMOTE, Random Forest)..." onkeyup="filterPapers();">
            </div>

            <div class="papers-table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Title of Paper</th>
                            <th>Authors</th>
                            <th>Year</th>
                            <th>Technique Used</th>
                            <th>Accuracy / Result</th>
                            <th>Dataset</th>
                            <th>DOI Link</th>
                        </tr>
                    </thead>
                    <tbody id="papersTableBody">
                        <!-- 30 papers rendered dynamically -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const allPapers = {{ all_papers | tojson }};

        function renderPapersTable(papers) {
            const tbody = document.getElementById('papersTableBody');
            tbody.innerHTML = '';
            papers.forEach(p => {
                const doiLink = p['Source URL'] && p['Source URL'] !== 'N/A' 
                    ? `<a href="${p['Source URL']}" target="_blank" class="link-doi">View DOI 🔗</a>` 
                    : '<span style="color:var(--text-muted)">N/A</span>';

                tbody.innerHTML += `
                    <tr>
                        <td><strong>${p['Sr.No']}</strong></td>
                        <td style="max-width: 250px; font-weight:600; color:#e2e8f0;">${p['Title of the Paper']}</td>
                        <td style="max-width: 140px; color:var(--text-muted); font-size:0.82rem;">${p['Authors']}</td>
                        <td><span class="badge" style="background:rgba(99, 102, 241, 0.15); color:#a5b4fc;">${p['Year']}</span></td>
                        <td style="max-width: 220px; font-size:0.82rem;">${p['Technique Used']}</td>
                        <td style="max-width: 200px; font-weight:600; color:#34d399; font-size:0.82rem;">${p['Accuracy / Main Result']}</td>
                        <td style="max-width: 150px; font-size:0.8rem; color:var(--text-muted);">${p['Dataset Used']}</td>
                        <td>${doiLink}</td>
                    </tr>
                `;
            });
        }

        function filterPapers() {
            const q = document.getElementById('searchInput').value.toLowerCase();
            const filtered = allPapers.filter(p => 
                p['Title of the Paper'].toLowerCase().includes(q) ||
                p['Authors'].toLowerCase().includes(q) ||
                p['Technique Used'].toLowerCase().includes(q) ||
                p['Accuracy / Main Result'].toLowerCase().includes(q)
            );
            renderPapersTable(filtered);
        }

        async function fetchPrediction() {
            const payload = {
                year: parseInt(document.getElementById('year').value),
                has_logistic_regression: parseInt(document.getElementById('has_logistic_regression').value),
                has_random_forest: parseInt(document.getElementById('has_random_forest').value),
                has_xgboost: parseInt(document.getElementById('has_xgboost').value),
                has_deep_learning: parseInt(document.getElementById('has_deep_learning').value),
                has_imbalance_handling: parseInt(document.getElementById('has_imbalance_handling').value),
                has_explainability: parseInt(document.getElementById('has_explainability').value),
                has_feature_selection: parseInt(document.getElementById('has_feature_selection').value),
                is_empirical_study: 1,
                is_ibm_telco_dataset: 1
            };

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.status !== 'success') return;

            const allResults = data.all_models;
            const metrics = data.metrics;
            const bestModel = data.best_model_name;

            const resGrid = document.getElementById('resultsGrid');
            resGrid.innerHTML = '';

            let maxProb = 0;
            for (const [name, res] of Object.entries(allResults)) {
                const prob = res.probability_percent;
                if (prob > maxProb) maxProb = prob;
                const isHighAcc = res.prediction === 1;

                resGrid.innerHTML += `
                    <div class="res-card ${name === bestModel ? 'active-best' : ''}">
                        <div class="m-title">${name} ${name === bestModel ? '🏆' : ''}</div>
                        <div class="m-prob" style="color: ${isHighAcc ? '#34d399' : '#f87171'};">${prob.toFixed(1)}%</div>
                        <div class="badge ${isHighAcc ? 'badge-high' : 'badge-low'}">
                            ${isHighAcc ? 'HIGH ACCURACY (≥85%)' : 'BASELINE ACCURACY (<85%)'}
                        </div>
                    </div>
                `;
            }

            const benchBody = document.getElementById('benchmarkTableBody');
            benchBody.innerHTML = '';
            for (const [name, m] of Object.entries(metrics)) {
                const isBest = name === bestModel;
                benchBody.innerHTML += `
                    <tr style="${isBest ? 'background: rgba(99, 102, 241, 0.15); color: #a5b4fc;' : ''}">
                        <td>${name} ${isBest ? '🏆' : ''}</td>
                        <td>${(m['Accuracy'] * 100).toFixed(1)}%</td>
                        <td>${m['Precision'].toFixed(3)}</td>
                        <td>${m['Recall'].toFixed(3)}</td>
                        <td>${m['F1-Score'].toFixed(3)}</td>
                        <td>${(m['CV Accuracy'] * 100).toFixed(1)}%</td>
                        <td>${m['ROC-AUC'].toFixed(3)}</td>
                    </tr>
                `;
            }

            const insightText = document.getElementById('insightText');
            if (maxProb > 75) {
                insightText.innerHTML = `<strong>HIGH PERFORMANCE METHODOLOGY:</strong> The selected features (SMOTE imbalance handling + Feature Selection + Ensemble Models) align with high-performing literature studies (Accuracy ≥ 85%).`;
            } else {
                insightText.innerHTML = `<strong>BASELINE METHODOLOGY:</strong> Standard baseline study without advanced class-imbalance oversampling or ensemble boosting. Recommend adding SMOTE and XGBoost for higher accuracy.`;
            }
        }

        document.getElementById('methodologyForm').addEventListener('submit', (e) => {
            e.preventDefault();
            fetchPrediction();
        });

        // Initialize table and predictions on load
        renderPapersTable(allPapers);
        fetchPrediction();
    </script>
</body>
</html>
"""

@app.route('/')
@app.route('/index')
@app.route('/index.py')
@app.route('/api/index')
def home():
    return render_template_string(HTML_TEMPLATE, all_papers=all_papers)

@app.route('/api/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

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

        all_models_results = {}
        for name, m in models.items():
            if name == 'Logistic Regression' and scaler is not None:
                inp_df = scaler.transform(df)
            else:
                inp_df = df

            prob = m.predict_proba(inp_df)[0][1]
            pred = m.predict(inp_df)[0]

            all_models_results[name] = {
                "prediction": int(pred),
                "high_performance_prob": float(prob),
                "probability_percent": round(float(prob) * 100, 2)
            }

        return jsonify({
            "status": "success",
            "best_model_name": best_model_name,
            "all_models": all_models_results,
            "metrics": metrics_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print("\nStarting 30 Research Papers Churn Dashboard Web Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
