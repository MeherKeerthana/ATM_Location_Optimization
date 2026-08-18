import sys

def verify_environment():
    print("=== Environment Verification ===")
    packages = {
        'flask': 'Flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'pulp': 'pulp',
        'xgboost': 'xgboost',
        'shap': 'shap',
        'reportlab': 'reportlab',
        'openpyxl': 'openpyxl'
    }
    all_imports_ok = True
    for module_name, pip_name in packages.items():
        try:
            __import__(module_name)
            print(f"  [OK] {pip_name} imported successfully.")
        except ImportError as e:
            print(f"  [FAILED] {pip_name} failed to import! Error: {e}")
            all_imports_ok = False
            
    if not all_imports_ok:
        print("\nCRITICAL ERROR: One or more required packages are not installed or cannot be imported.")
        print("Please activate your virtual environment and run 'pip install -r requirements.txt'.\n")
        sys.exit(1)
        
    import pulp
    print("Checking PuLP CBC solver availability...")
    try:
        cbc = pulp.COIN_CMD(msg=False)
        cbc_available = cbc.available()
        
        default_solver = pulp.LpSolverDefault
        default_available = default_solver.available() if default_solver else False
        
        if cbc_available:
            print("  [OK] PuLP CBC solver is available.")
        elif default_available:
            print(f"  [OK] PuLP default solver ({default_solver.name}) is available.")
        else:
            print("\nWARNING: PuLP CBC solver is NOT available on this system!")
            print("This is a common install issue. Without a working solver, optimization calculations will fail.")
            print("Troubleshooting options:")
            print("  1. Try reinstalling pulp: 'pip install --force-reinstall pulp'")
            print("  2. Ensure your environment allows running system binaries.")
            print("  3. On Linux/macOS, you may need to install coinor-cbc package (e.g., 'sudo apt-get install coinor-cbc').\n")
    except Exception as e:
        print(f"  [FAILED] Error checking solver: {e}")
        
    print("Environment verification complete.\n")

# Import the actual models after verification to prevent import errors at the top of file
try:
    import pandas as pd
    import numpy as np
    from ml_engine import ATMMLModel, ATMOptimizer, ATMRiskClassifier
except ImportError:
    pass

def test_pipeline():
    print("=== Testing ML & Optimization Pipeline ===")
    
    # 1. Load data
    print("Loading data...")
    demographics = pd.read_csv('data/demographics.csv')
    own_atms = pd.read_csv('data/atms_own.csv')
    comp_atms = pd.read_csv('data/atms_competitor.csv')
    candidates = pd.read_csv('data/candidates.csv')
    
    print(f"Demographics shape: {demographics.shape}")
    print(f"Own ATMs shape: {own_atms.shape}")
    print(f"Competitor ATMs shape: {comp_atms.shape}")
    print(f"Candidates shape: {candidates.shape}")
    
    # 2. Test ML model training
    print("\nTesting ML Model Training...")
    model = ATMMLModel()
    r2_score = model.train(own_atms, comp_atms)
    print(f"Model trained successfully. Best Model R^2 score: {r2_score:.4f}")
    print("Model Performance Comparison:")
    for m_name, m_val in model.metrics.items():
        print(f"  - {m_name:16s} -> R2: {m_val['r2_score']:.4f}, MAE: {m_val['mae']:.2f}, RMSE: {m_val['rmse']:.2f}, Accuracy: {m_val['accuracy']:.2f}%")
        
    # Single-model policy: ATMMLModel now trains XGBoost only (matches
    # ATMRiskClassifier below), so there's no cross-model R2 gap to check
    # anymore -- just confirm XGBoost itself trained to a sane fit and that
    # avg_income importance is in the expected range.
    assert set(model.metrics.keys()) == {'XGBoost'}, f"Expected XGBoost-only model dict, got {list(model.metrics.keys())}"
    xgb_r2 = model.metrics['XGBoost']['r2_score']
    print(f"  - XGBoost R2 score: {xgb_r2:.4f}")
    assert xgb_r2 >= 0.70, f"Expected XGBoost R2 >= 0.70, got {xgb_r2:.4f}"
    
    xgb_inc_imp = model.feature_importances_by_model['XGBoost']['avg_income']
    print(f"  - avg_income Importance: XGBoost={xgb_inc_imp*100:.2f}%")
    assert 0.30 <= xgb_inc_imp <= 0.50, f"Expected XGBoost avg_income importance between 30% and 50%, got {xgb_inc_imp*100:.2f}%"
    
    print("\nBest Model Feature Importances:")
    for feat, imp in model.feature_importances_.items():
        print(f"  - {feat}: {imp:.4f}")
        
    # 3. Test Candidate predictions
    print("\nTesting Candidate Predictions...")
    predicted_candidates = model.predict_candidates(candidates, own_atms, comp_atms)
    print(f"Predicted candidates count: {len(predicted_candidates)}")
    assert 'predicted_daily_transactions' in predicted_candidates.columns, "Predictions missing column"
    assert 'roi_index' in predicted_candidates.columns, "Predictions missing ROI index column"
    
    print("Candidate predictions preview:")
    print(predicted_candidates[['candidate_id', 'predicted_daily_transactions', 'roi_index']].head())
    
    # 4. Test Optimizer
    print("\nTesting Spatial Optimizer...")
    optimizer = ATMOptimizer(demographics, own_atms, comp_atms)
    
    # MCLP
    print("Solving MCLP (k=3, R=1.0km)...")
    selected_mclp = optimizer.solve_mclp(predicted_candidates, k=3, radius_km=1.0)
    print(f"MCLP selected {len(selected_mclp)} candidate(s):")
    print(selected_mclp[['candidate_id', 'name', 'latitude', 'longitude']])
    assert len(selected_mclp) <= 3, "Selected count exceeds budget k"
    
    # p-Median
    print("Solving p-Median (k=3)...")
    selected_pmed = optimizer.solve_p_median(predicted_candidates, k=3)
    print(f"p-Median selected {len(selected_pmed)} candidate(s):")
    print(selected_pmed[['candidate_id', 'name', 'latitude', 'longitude']])
    assert len(selected_pmed) <= 3, "Selected count exceeds budget k"
    
    # ML-Revenue Maximizer
    print("Solving ML-Revenue Maximizer (k=3, min_dist=0.5km)...")
    selected_rev = optimizer.solve_revenue_maximizer(predicted_candidates, k=3, min_dist_km=0.5)
    print(f"ML-Revenue Maximizer selected {len(selected_rev)} candidate(s):")
    print(selected_rev[['candidate_id', 'name', 'latitude', 'longitude']])
    assert len(selected_rev) <= 3, "Selected count exceeds budget k"
    
    # Metrics
    print("\nCalculating metrics lift for MCLP selection...")
    metrics = optimizer.calculate_metrics_impact(selected_mclp, radius_km=1.0)
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
        
    # 5. Test Risk Classifier
    print("\nTesting Risk Classifier (ATMRiskClassifier)...")
    risk_clf = ATMRiskClassifier()
    risk_clf.train(own_atms, comp_atms, candidates)
    
    xgb_metrics = risk_clf.metrics['XGBoost']
    roc_auc = xgb_metrics['roc_auc']
    pr_auc = xgb_metrics['pr_auc']
    cm = risk_clf.confusion_matrix_data
    wf = risk_clf.get_shap_waterfall('ATM_001')
    base_prob = round(wf['base_value'] * 100.0, 1)
    
    shap_sum = risk_clf.get_shap_summary()
    sorted_shap = sorted(shap_sum.items(), key=lambda x: x[1], reverse=True)
    top1_feat, top1_val = sorted_shap[0]
    top2_feat, top2_val = sorted_shap[1]
    shap_ratio = top1_val / top2_val if top2_val > 0 else 0.0
    err_rate = (cm['fp'] + cm['fn']) / len(own_atms) * 100.0
    
    print(f"  - Risk Classifier ROC AUC: {roc_auc:.1f}% | PR-AUC: {pr_auc:.1f}%")
    print(f"  - Risk Classifier Confusion Matrix: TN:{cm['tn']}/FP:{cm['fp']}/FN:{cm['fn']}/TP:{cm['tp']} (Error Rate: {err_rate:.2f}%)")
    print(f"  - Risk Classifier Waterfall Base Probability: {base_prob:.1f}%")
    print(f"  - SHAP Dominance Ratio ({top1_feat} / {top2_feat}): {shap_ratio:.2f}x")
    
    assert roc_auc >= 95.0, f"Expected ROC AUC >= 95%, got {roc_auc:.1f}%"
    assert pr_auc >= 95.0, f"Expected PR-AUC >= 95%, got {pr_auc:.1f}%"
    assert (cm['fp'] + cm['fn']) / len(own_atms) <= 0.10, f"Expected error rate <= 10%, got {err_rate:.2f}%"
    assert shap_ratio <= 1.5, f"Expected SHAP dominance ratio <= 1.5x, got {shap_ratio:.2f}x"
    assert (cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']) == len(own_atms), f"Confusion matrix total should be {len(own_atms)}, got {cm}"
    assert 30.0 <= base_prob <= 60.0, f"Expected base probability between 30% and 60%, got {base_prob}%"

    print("\n=== All Tests Passed Successfully ===")


if __name__ == '__main__':
    verify_environment()
    test_pipeline()