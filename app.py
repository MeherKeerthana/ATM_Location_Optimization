from flask import Flask, jsonify, request, send_from_directory, send_file
import pandas as pd
import numpy as np
import os
import json
import shutil
import traceback
from data_generator import run_data_generation
from ml_engine import ATMMLModel, ATMOptimizer, ATMRiskClassifier
from download_assets import download_web_assets
from report_export import generate_pdf_report, generate_excel_report

# Frontend source files live alongside app.py; Flask serves everything out of
# static/ via send_from_directory, so they need to be copied there. Without
# this, '/' 404s until someone copies index.html/app.css/app.js by hand.
FRONTEND_SOURCE_FILES = ['index.html', 'app.css', 'app.js']

def ensure_static_files():
    os.makedirs('static', exist_ok=True)
    for fname in FRONTEND_SOURCE_FILES:
        if os.path.exists(fname):
            try:
                shutil.copy2(fname, os.path.join('static', fname))
            except Exception as e:
                print(f"WARNING: could not copy {fname} into static/: {e}")

app = Flask(__name__, static_folder='static')
# Ensure JSON responses preserve insertion order for nested segment dicts
app.config['JSON_SORT_KEYS'] = False
try:
    # For Flask versions that expose the JSON provider
    app.json.sort_keys = False
except Exception:
    pass

# Global variables for data
DEMOGRAPHICS_DF = None
OWN_ATMS_DF = None
COMP_ATMS_DF = None
CANDIDATES_DF = None
TX_LOGS_DF = None

ML_MODEL = None
OPTIMIZER = None
RISK_CLASSIFIER = None
STARTUP_ERROR = None
ASSET_DOWNLOAD_ERROR = None

def load_data():
    global DEMOGRAPHICS_DF, OWN_ATMS_DF, COMP_ATMS_DF, CANDIDATES_DF, TX_LOGS_DF, ML_MODEL, OPTIMIZER, RISK_CLASSIFIER, STARTUP_ERROR, ASSET_DOWNLOAD_ERROR
    
    # Make sure the frontend files (index.html/app.css/app.js) are in static/
    # regardless of what happens below, so '/' can always be served.
    ensure_static_files()
    
    # Verify and download required third-party web assets (leaflet, chart.js,
    # lucide). A failure here only breaks map/chart rendering in the browser
    # -- it has nothing to do with the data/ML pipeline below, so we log it
    # and keep going instead of aborting the whole startup (that used to
    # take down every /api/* endpoint over a CDN hiccup unrelated to them).
    print("Checking required web assets...")
    try:
        download_web_assets(force=False)
        print("Web assets verification complete.")
    except Exception as e:
        ASSET_DOWNLOAD_ERROR = f"Asset download failed: {str(e)}"
        print("\n" + "="*80)
        print("WARNING: Frontend asset download failed. Maps/charts in the")
        print("dashboard may be broken, but the data/ML/optimization API")
        print("will still initialize normally.")
        print(traceback.format_exc())
        print("="*80 + "\n")

    try:
        # Generate data if it doesn't exist
        if not os.path.exists('data/demographics.csv'):
            print("Data files not found. Generating synthetic data...")
            run_data_generation()
            
        # Load dataframes
        DEMOGRAPHICS_DF = pd.read_csv('data/demographics.csv')
        OWN_ATMS_DF = pd.read_csv('data/atms_own.csv')
        COMP_ATMS_DF = pd.read_csv('data/atms_competitor.csv')
        CANDIDATES_DF = pd.read_csv('data/candidates.csv')
        TX_LOGS_DF = pd.read_csv('data/transaction_logs.csv')
        # Parse timestamps once here rather than on every /api/analytics
        # request -- the column doesn't change after load.
        TX_LOGS_DF['datetime'] = pd.to_datetime(TX_LOGS_DF['timestamp'])
        
        print(f"Data loaded successfully:")
        print(f"  - Demographics: {len(DEMOGRAPHICS_DF)} grid cells")
        print(f"  - Own ATMs: {len(OWN_ATMS_DF)} sites")
        print(f"  - Competitor ATMs: {len(COMP_ATMS_DF)} sites")
        print(f"  - Candidates: {len(CANDIDATES_DF)} locations")
        print(f"  - Transactions: {len(TX_LOGS_DF)} records")
        
        # Train ML Model
        ML_MODEL = ATMMLModel()
        model_label = " / ".join(ML_MODEL.models.keys())
        print(f"Training demand prediction model ({model_label})...")
        score = ML_MODEL.train(OWN_ATMS_DF, COMP_ATMS_DF)
        print(f"Model trained. R^2 score: {score:.4f}")
        
        # Initialize and train Risk Classifier BEFORE candidate predictions,
        # so its risk scores can be combined into predict_candidates below
        # (Option B: expected_value_score = predicted_transactions * (1 - risk)).
        RISK_CLASSIFIER = ATMRiskClassifier()
        risk_model_label = " / ".join(RISK_CLASSIFIER.models.keys())
        print(f"Training risk prediction model ({risk_model_label} Classifier)...")
        RISK_CLASSIFIER.train(OWN_ATMS_DF, COMP_ATMS_DF, CANDIDATES_DF)
        print("Risk classifier trained successfully.")
        
        # Run predictions on candidates on startup (now includes risk_probability
        # and expected_value_score alongside predicted_daily_transactions/roi_index)
        CANDIDATES_DF = ML_MODEL.predict_candidates(CANDIDATES_DF, OWN_ATMS_DF, COMP_ATMS_DF, risk_classifier=RISK_CLASSIFIER)
        
        # Initialize Optimizer
        OPTIMIZER = ATMOptimizer(DEMOGRAPHICS_DF, OWN_ATMS_DF, COMP_ATMS_DF)
        
    except Exception as e:
        STARTUP_ERROR = f"Startup initialization failed: {str(e)}"
        print("\n" + "="*80)
        print("CRITICAL ERROR DURING PIPELINE INITIALIZATION:")
        print(traceback.format_exc())
        print("="*80 + "\n")

# Endpoint: Serve index.html
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Endpoint: Serve app.css from static folder to prevent 404 at root URL
@app.route('/app.css')
def css():
    return send_from_directory('static', 'app.css')

# Endpoint: Serve app.js from static folder to prevent 404 at root URL
@app.route('/app.js')
def js():
    return send_from_directory('static', 'app.js')

# Endpoint: Get spatial data
@app.route('/api/data', methods=['GET'])
def get_data():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
        
    demographics_json = DEMOGRAPHICS_DF.to_dict(orient='records')
    own_atms_json = OWN_ATMS_DF.to_dict(orient='records')
    comp_atms_json = COMP_ATMS_DF.to_dict(orient='records')
    
    # Send predictions for all models grouped by model name
    candidates_by_model = {}
    for name in ML_MODEL.models.keys():
        pred_df = ML_MODEL.predict_candidates(CANDIDATES_DF, OWN_ATMS_DF, COMP_ATMS_DF, model_name=name, risk_classifier=RISK_CLASSIFIER)
        candidates_by_model[name] = pred_df.to_dict(orient='records')
        
    # Default candidate list uses best model predictions
    default_candidates = candidates_by_model[ML_MODEL.best_model_name]
    
    return jsonify({
        'demographics': demographics_json,
        'own_atms': own_atms_json,
        'competitor_atms': comp_atms_json,
        'candidates': default_candidates,
        'candidates_by_model': candidates_by_model
    })

# Endpoint: Get analytics data
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
        
    # 1. Hourly trend (datetime column parsed once in load_data())
    hourly_counts = TX_LOGS_DF.groupby(TX_LOGS_DF['datetime'].dt.hour).size().reset_index(name='count')
    hourly_trend = {int(r['datetime']): int(r['count']) for _, r in hourly_counts.iterrows()}
    
    # 2. Day of week trend
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = TX_LOGS_DF.groupby(TX_LOGS_DF['datetime'].dt.weekday).size().reset_index(name='count')
    dow_trend = {dow_names[int(r['datetime'])]: int(r['count']) for _, r in dow_counts.iterrows()}
    
    # 3. Transaction type distribution
    tx_types = TX_LOGS_DF.groupby('transaction_type').size().to_dict()
    tx_types = {k: int(v) for k, v in tx_types.items()}
    
    # 4. Card type distribution
    card_types = TX_LOGS_DF.groupby('card_type').size().to_dict()
    card_types = {k: int(v) for k, v in card_types.items()}
    
    # 5. Status distribution
    status_counts = TX_LOGS_DF.groupby('status').size().to_dict()
    status_counts = {k: int(v) for k, v in status_counts.items()}
    
    # 6. Model metrics and explainability (PDP)
    model_metrics = {
        'best_model_name': ML_MODEL.best_model_name,
        'metrics': ML_MODEL.metrics,
        'feature_importances': ML_MODEL.feature_importances_by_model,
        'pdp_data': ML_MODEL.pdp_data,
        'insights': ML_MODEL.insights
    }
    
    # 7. Predicted vs Actual transactions -- genuine held-out predictions only.
    # These ATMs were withheld from training entirely, so this reflects real
    # generalization performance and lines up with the R^2 in model_metrics,
    # rather than re-predicting on data the model has already memorized.
    held_out_preds = ML_MODEL.test_predictions_by_model[ML_MODEL.best_model_name]
    predicted_vs_actual = []
    for atm_id, actual_val, pred_val in zip(ML_MODEL.test_atm_ids, ML_MODEL.test_actual, held_out_preds):
        predicted_vs_actual.append({
            'atm_id': atm_id,
            'actual': int(actual_val),
            'predicted': int(round(pred_val))
        })
        
    # 8. Zone-wise metrics (avg transactions, rent, ROI)
    zone_groups = OWN_ATMS_DF.groupby('zone_name').agg({
        'avg_daily_transactions': 'mean',
        'rent_cost': 'mean'
    }).reset_index()
    
    zone_wise_metrics = []
    for _, row in zone_groups.iterrows():
        avg_tx = row['avg_daily_transactions']
        avg_rent = row['rent_cost']
        avg_roi = (avg_tx * 15.35 * 30) / avg_rent if avg_rent > 0 else 0
        zone_wise_metrics.append({
            'zone': row['zone_name'],
            'avg_transactions': round(avg_tx, 1),
            'avg_rent': round(avg_rent, 1),
            'avg_roi': round(avg_roi, 2)
        })
        
    return jsonify({
        'hourly_trend': hourly_trend,
        'dow_trend': dow_trend,
        'transaction_types': tx_types,
        'card_types': card_types,
        'status_counts': status_counts,
        'model_metrics': model_metrics,
        'predicted_vs_actual': predicted_vs_actual,
        'zone_wise_metrics': zone_wise_metrics
    })

def apply_ml_prefilter(candidates_df, k, min_multiplier=5, min_fraction=0.3):
    """
    Stage 1 of a two-stage placement pipeline for MCLP / p-median:
    rank ALL candidates by their ML expected-value score and keep a
    generous shortlist before handing them to the spatial optimizer.

    This gives ML a real, direct role -- every site MCLP/p-median can even
    consider must first clear this financial bar -- rather than the old
    setup where ML only nudged between near-ties (blend_weight) after
    spatial optimization ran over the full candidate pool.

    The shortlist is deliberately generous (max(k*5, 30% of all
    candidates), not just the top k) so stage 2 still has genuinely
    different real coverage/distance options to choose between -- if we
    only kept the top k by ML score, MCLP/p-median would have nothing
    left to optimize and would just rubber-stamp the ML ranking.

    Trade-off worth knowing: this means these two methods no longer solve
    "best coverage across every possible site" -- they solve "best
    coverage among financially strong sites". A location that closes a
    real coverage gap but has a middling predicted revenue can be
    filtered out here before MCLP/p-median ever see it.
    """
    score_col = 'expected_value_score' if 'expected_value_score' in candidates_df.columns else 'predicted_daily_transactions'
    shortlist_n = max(k * min_multiplier, int(np.ceil(len(candidates_df) * min_fraction)))
    shortlist_n = min(shortlist_n, len(candidates_df))
    shortlisted = candidates_df.sort_values(by=score_col, ascending=False).head(shortlist_n).reset_index(drop=True)
    return shortlisted, score_col, shortlist_n


# Endpoint: Run Location Optimization
@app.route('/api/optimize', methods=['POST'])
def optimize_placement():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
        
    params = request.get_json() or {}
    k = int(params.get('k', 3))
    radius = float(params.get('radius', 1.0))
    method = params.get('method', 'mclp') # 'mclp', 'p-median', or 'revenue'
    model_name = params.get('model_name', ML_MODEL.best_model_name)
    objective = params.get('objective', 'transactions')  # only used when method == 'revenue': 'transactions', 'net_profit', 'roi', or 'expected_value'
    # ML blend for MCLP/p-median: how much the risk-adjusted expected_value_score
    # nudges site selection among candidates with similar real coverage/distance
    # value, WITHIN the ML-prefiltered shortlist (see apply_ml_prefilter above).
    # Default 0.05 was tuned against the real (non-toy) generated dataset: it
    # moves selection toward meaningfully higher-ML-scored sites while leaving
    # the actual coverage/distance metric changed by well under 1% -- i.e. it
    # breaks near-ties in the ML model's favor rather than overriding genuine
    # spatial optimization. Client can override or disable (blend_weight=0)
    # via the request body.
    use_ml_blend = bool(params.get('use_ml_blend', True))
    blend_weight = float(params.get('blend_weight', 0.05)) if use_ml_blend else 0.0
    
    # Normalize an unrecognized method to 'mclp' up front, so every
    # downstream use of `method` (branch selection AND the summary text)
    # reflects what was actually solved instead of echoing back a typo or
    # an invalid client-supplied string.
    VALID_METHODS = {'mclp', 'p-median', 'revenue'}
    if method not in VALID_METHODS:
        print(f"WARNING: unrecognized optimize method '{method}', defaulting to 'mclp'")
        method = 'mclp'
    
    print(f"Running optimization request: k={k}, radius={radius}km, method={method}, model={model_name}, objective={objective}")
    
    # Compute candidate predictions for the active model
    active_candidates_df = ML_MODEL.predict_candidates(CANDIDATES_DF, OWN_ATMS_DF, COMP_ATMS_DF, model_name=model_name, risk_classifier=RISK_CLASSIFIER)
    
    # expected_value_score (predicted transactions discounted by predicted
    # underperformance risk) is what MCLP/p-median get nudged toward -- it's
    # only present when the risk classifier successfully scored candidates,
    # so we fall back to no ML influence rather than erroring if it's missing.
    ml_scores = None
    if 'expected_value_score' in active_candidates_df.columns:
        ml_scores = dict(zip(active_candidates_df['candidate_id'], active_candidates_df['expected_value_score']))

    prefilter_info = None
    if method in ('mclp', 'p-median'):
        # Stage 1: ML pre-filter down to a shortlist of financially strong
        # candidates. Stage 2 (below) runs the real spatial optimization
        # only over that shortlist.
        shortlisted_df, score_col, shortlist_n = apply_ml_prefilter(active_candidates_df, k)
        prefilter_info = {
            'total_candidates': len(active_candidates_df),
            'shortlisted_candidates': shortlist_n,
            'ranked_by': score_col
        }
        optimizer_input_df = shortlisted_df
    else:
        optimizer_input_df = active_candidates_df

    if method == 'p-median':
        selected_df = OPTIMIZER.solve_p_median(optimizer_input_df, k, ml_scores=ml_scores, blend_weight=blend_weight)
    elif method == 'revenue':
        # radius parameter serves as the minimum separation distance (km)
        selected_df = OPTIMIZER.solve_revenue_maximizer(optimizer_input_df, k, min_dist_km=radius, objective=objective)
    else:  # Default to MCLP
        selected_df = OPTIMIZER.solve_mclp(optimizer_input_df, k, radius_km=radius, ml_scores=ml_scores, blend_weight=blend_weight)
        
    metrics = OPTIMIZER.calculate_metrics_impact(selected_df, radius_km=radius)
    
    # Add predicted transaction, ROI, and payback period details to selected items
    selected_list = selected_df.to_dict(orient='records')
    for item in selected_list:
        match = active_candidates_df[active_candidates_df['candidate_id'] == item['candidate_id']]
        if len(match) > 0:
            item['predicted_daily_transactions'] = int(match.iloc[0]['predicted_daily_transactions'])
            item['roi_index'] = float(match.iloc[0]['roi_index'])
            item['payback_period'] = float(match.iloc[0]['payback_period'])
            
    summary_text = (
        f"Optimized placement using {method.upper()} solved successfully. "
        f"Selected {len(selected_list)} out of {len(CANDIDATES_DF)} candidate sites. "
        f"Population coverage within {radius}km increased from {metrics['coverage_before']}% to {metrics['coverage_after']}% "
        f"(a net gain of +{metrics['coverage_increase']}%). "
        f"Average customer travel distance reduced by {metrics['avg_dist_reduction_km']} km (new average: {metrics['avg_dist_after_km']} km)."
    )
    if prefilter_info:
        summary_text = (
            f"Stage 1: ML pre-filtered {prefilter_info['total_candidates']} candidates down to the "
            f"{prefilter_info['shortlisted_candidates']} strongest sites by {prefilter_info['ranked_by']}. "
            f"Stage 2: {summary_text}"
        )
    if len(selected_list) < k:
        summary_text += (
            f" Only {len(selected_list)} of the requested {k} sites were selected because "
            f"the remaining candidates would add no further coverage gain within the {radius}km radius."
        )
    
    return jsonify({
        'selected_candidates': selected_list,
        'metrics': metrics,
        'summary': summary_text,
        'objective': objective if method == 'revenue' else None,
        'ml_prefilter': prefilter_info
    })


# Endpoint: Simulate removing an existing own ATM from the network
@app.route('/api/simulate-removal', methods=['POST'])
def simulate_removal():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500

    params = request.get_json() or {}
    atm_id = params.get('atm_id')
    radius = float(params.get('radius', 1.0))

    if not atm_id:
        return jsonify({'error': 'Missing atm_id parameter'}), 400

    try:
        impact = OPTIMIZER.calculate_removal_impact(atm_id, radius_km=radius)
        return jsonify(impact)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print("Error simulating ATM removal:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Endpoint: Get Risk Analytics data
@app.route('/api/risk-analytics', methods=['GET'])
def get_risk_analytics():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
        
    try:
        # Calculate risk scores on existing own ATMs. RISK_CLASSIFIER.X_all is
        # [own ATMs rows] followed by [candidate rows] (see ATMRiskClassifier.train),
        # so the split point is len(OWN_ATMS_DF), not a hardcoded constant --
        # this stays correct even if data_generator's own-ATM count changes.
        n_own = len(OWN_ATMS_DF)
        own_probs = RISK_CLASSIFIER.best_model.predict_proba(RISK_CLASSIFIER.X_all.iloc[:n_own])[:, 1]
        
        # Buckets: Low (<=0.3), Med (0.3-0.7), High (>0.7)
        low_count = int(np.sum(own_probs <= 0.3))
        med_count = int(np.sum((own_probs > 0.3) & (own_probs <= 0.7)))
        high_count = int(np.sum(own_probs > 0.7))
        
        risk_distribution = {
            'Low': low_count,
            'Medium': med_count,
            'High': high_count
        }
        
        # Calculate headline business summary metrics from candidate predictions and risk classifier
        cand_pred_df = ML_MODEL.predict_candidates(CANDIDATES_DF, OWN_ATMS_DF, COMP_ATMS_DF, risk_classifier=RISK_CLASSIFIER)
        strong_cands = cand_pred_df[cand_pred_df['roi_index'] >= 1.0]
        n_strong = int(len(strong_cands))
        combined_revenue = float(strong_cands['monthly_revenue'].sum()) if n_strong > 0 else 0.0
        z_flagged = int(np.sum(own_probs >= 0.5))
        high_risk_count = int(np.sum(own_probs > 0.7))

        headline_summary = {
            'strong_candidate_count': n_strong,
            'combined_monthly_revenue': round(combined_revenue, 2),
            'watchlist_site_count': z_flagged,
            'high_risk_site_count': high_risk_count
        }

        own_risk_list = []
        for i in range(n_own):
            row = OWN_ATMS_DF.iloc[i]
            prob = float(own_probs[i])
            wf = RISK_CLASSIFIER.get_shap_waterfall(row['atm_id'])
            own_risk_list.append({
                'id': row['atm_id'],
                'name': f"ATM {row['atm_id'].split('_')[1]} - {row['zone_name']}",
                'zone': row['zone_name'],
                'probability': round(prob, 3),
                'risk_tier': 'High' if prob > 0.7 else ('Medium' if prob > 0.3 else 'Low'),
                'top_diagnosis': wf.get('top_diagnosis', '')
            })
            
        cand_probs = RISK_CLASSIFIER.best_model.predict_proba(RISK_CLASSIFIER.X_all.iloc[n_own:])[:, 1]
        cand_risk_list = []
        for i in range(len(CANDIDATES_DF)):
            row = CANDIDATES_DF.iloc[i]
            prob = float(cand_probs[i])
            wf = RISK_CLASSIFIER.get_shap_waterfall(row['candidate_id'])
            cand_risk_list.append({
                'id': row['candidate_id'],
                'name': row['name'],
                'zone': row['zone_name'],
                'probability': round(prob, 3),
                'risk_tier': 'High' if prob > 0.7 else ('Medium' if prob > 0.3 else 'Low'),
                'top_diagnosis': wf.get('top_diagnosis', '')
            })
            
        # Segment Analysis (existing own ATMs)
        temp_df = OWN_ATMS_DF.copy()
        temp_df['risk_probability'] = own_probs
        
        # By Zone
        zone_data = temp_df.groupby('zone_name')['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        segment_by_zone = {k: {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])} for k, v in zone_data.items()}
        
        # By Site Type
        site_type_data = temp_df.groupby('site_type')['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        segment_by_site_type = {k: {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])} for k, v in site_type_data.items()}
        
        # By Area Type
        area_type_data = temp_df.groupby('area_type')['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        segment_by_area_type = {k: {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])} for k, v in area_type_data.items()}
        
        # By Months in Service (tenure)
        def get_tenure_bin(m):
            if m <= 12: return 'New (<=1yr)'
            elif m <= 36: return 'Mid-Tenure (1-3yr)'
            elif m <= 72: return 'Mature (3-6yr)'
            else: return 'Legacy (>6yr)'
        temp_df['tenure_bin'] = temp_df['months_in_service'].apply(get_tenure_bin)
        tenure_order = ['New (<=1yr)', 'Mid-Tenure (1-3yr)', 'Mature (3-6yr)', 'Legacy (>6yr)']
        temp_df['tenure_bin'] = pd.Categorical(temp_df['tenure_bin'], categories=tenure_order, ordered=True)
        tenure_data = temp_df.groupby('tenure_bin', observed=True)['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        # Preserve logical ordering by iterating the defined category list
        segment_by_tenure = {}
        for cat in tenure_order:
            if cat in tenure_data:
                v = tenure_data[cat]
                segment_by_tenure[cat] = {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])}
        
        # Top 10 Zones by Risk
        top_10_zones = sorted(segment_by_zone.items(), key=lambda x: x[1]['risk_rate'], reverse=True)[:10]
        top_10_zones = {k: v for k, v in top_10_zones}
        
        # Diagnostic Curve details:
        # Learning Curve
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            RISK_CLASSIFIER.best_model, RISK_CLASSIFIER.X_train, RISK_CLASSIFIER.y_train,
            cv=3, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', random_state=42
        )
        train_mean = np.mean(train_scores, axis=1).tolist()
        test_mean = np.mean(test_scores, axis=1).tolist()
        learning_curve_data = {
            'sizes': [int(s) for s in train_sizes],
            'train_scores': [round(float(s), 4) for s in train_mean],
            'test_scores': [round(float(s), 4) for s in test_mean]
        }
        
        # Calibration Curve
        from sklearn.calibration import calibration_curve
        xgb_test_probs = RISK_CLASSIFIER.best_model.predict_proba(RISK_CLASSIFIER.X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(RISK_CLASSIFIER.y_test, xgb_test_probs, n_bins=5)
        calibration_curve_data = {
            'prob_true': [round(float(x), 4) for x in prob_true],
            'prob_pred': [round(float(x), 4) for x in prob_pred]
        }
        
        # Heatmap, target distribution
        corr_matrix = RISK_CLASSIFIER.X_all.corr().round(2).to_dict()
        y_all = RISK_CLASSIFIER.y_train.tolist() + RISK_CLASSIFIER.y_test.tolist()
        target_distribution = {
            'Performing': int(np.sum(np.array(y_all) == 0)),
            'Underperforming': int(np.sum(np.array(y_all) == 1))
        }
        
        # Site Insights: site_type x area_type
        pivot = temp_df.groupby(['site_type', 'area_type'])['risk_probability'].agg(['mean', 'count']).reset_index()
        heatmap_data = []
        for _, r in pivot.iterrows():
            heatmap_data.append({
                'site_type': r['site_type'],
                'area_type': r['area_type'],
                'risk_rate': round(float(r['mean']) * 100.0, 2),
                'count': int(r['count'])
            })
            
        # By Uptime Bracket
        def get_uptime_bracket(up):
            if up < 90: return 'Critical (<90%)'
            elif up < 95: return 'Substandard (90-95%)'
            elif up < 98: return 'Target (95-98%)'
            else: return 'Excellent (>=98%)'
        temp_df['uptime_bracket'] = temp_df['uptime_pct'].apply(get_uptime_bracket)
        uptime_order = ['Critical (<90%)', 'Substandard (90-95%)', 'Target (95-98%)', 'Excellent (>=98%)']
        temp_df['uptime_bracket'] = pd.Categorical(temp_df['uptime_bracket'], categories=uptime_order, ordered=True)
        uptime_data = temp_df.groupby('uptime_bracket', observed=True)['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        # Build ordered uptime segments using uptime_order
        segment_by_uptime = {}
        for cat in uptime_order:
            if cat in uptime_data:
                v = uptime_data[cat]
                segment_by_uptime[cat] = {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])}
        
        # At risk by site type
        at_risk_by_site_type = temp_df[temp_df['risk_probability'] > 0.7].groupby('site_type').size().to_dict()
        at_risk_by_site_type = {k: int(v) for k, v in at_risk_by_site_type.items()}
        
        # By Income Bracket
        def get_income_bracket(inc):
            if inc < 600000: return 'Low Income (<6L)'
            elif inc < 1200000: return 'Middle Income (6-12L)'
            elif inc < 2000000: return 'Upper-Middle (12-20L)'
            else: return 'High Income (>=20L)'
        temp_df['income_bracket'] = temp_df['avg_income'].apply(get_income_bracket)
        income_order = ['Low Income (<6L)', 'Middle Income (6-12L)', 'Upper-Middle (12-20L)', 'High Income (>=20L)']
        temp_df['income_bracket'] = pd.Categorical(temp_df['income_bracket'], categories=income_order, ordered=True)
        income_data = temp_df.groupby('income_bracket', observed=True)['risk_probability'].agg(['mean', 'count']).to_dict(orient='index')
        # Build ordered income segments using income_order
        segment_by_income = {}
        for cat in income_order:
            if cat in income_data:
                v = income_data[cat]
                segment_by_income[cat] = {'risk_rate': round(v['mean'] * 100.0, 2), 'count': int(v['count'])}
        
        return jsonify({
            'headline_summary': headline_summary,
            'risk_distribution': risk_distribution,
            'feature_importances': RISK_CLASSIFIER.feature_importances_by_model,
            'metrics': RISK_CLASSIFIER.metrics,
            'confusion_matrix': RISK_CLASSIFIER.confusion_matrix_data,
            'shap_summary': RISK_CLASSIFIER.get_shap_summary(),
            'learning_curve': learning_curve_data,
            'calibration_curve': calibration_curve_data,
            'correlation_matrix': corr_matrix,
            'target_distribution': target_distribution,
            'segment_by_zone': segment_by_zone,
            'segment_by_site_type': segment_by_site_type,
            'segment_by_area_type': segment_by_area_type,
            'segment_by_tenure': segment_by_tenure,
            'top_10_zones': top_10_zones,
            'segment_by_uptime': segment_by_uptime,
            'segment_by_income': segment_by_income,
            'at_risk_by_site_type': at_risk_by_site_type,
            'heatmap_data': heatmap_data,
            'own_risk_list': own_risk_list,
            'cand_risk_list': cand_risk_list
        })
        
    except Exception as e:
        print("Error serving risk analytics:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Endpoint: Get Single Site SHAP Waterfall plot data
@app.route('/api/risk-analytics/waterfall', methods=['GET'])
def get_risk_waterfall():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
        
    try:
        site_id = request.args.get('site_id')
        if not site_id:
            return jsonify({'error': 'Missing site_id parameter'}), 400
            
        waterfall_data = RISK_CLASSIFIER.get_shap_waterfall(site_id)
        return jsonify(waterfall_data), 200
    except Exception as e:
        print("Error serving risk waterfall:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Endpoint: Export a dashboard snapshot as a PDF report.
# The frontend posts a lightweight JSON snapshot of what's currently on
# screen (KPIs, last optimization run, last removal simulation, model
# metrics, risk summary) -- see report_export.py for the exact shape.
@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
    try:
        payload = request.get_json() or {}
        buffer = generate_pdf_report(payload)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='optiatm_network_report.pdf'
        )
    except Exception as e:
        print("Error generating PDF export:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Endpoint: Export a dashboard snapshot as a multi-sheet Excel workbook.
@app.route('/api/export/excel', methods=['POST'])
def export_excel():
    if STARTUP_ERROR:
        return jsonify({'error': STARTUP_ERROR}), 500
    try:
        payload = request.get_json() or {}
        buffer = generate_excel_report(payload)
        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='optiatm_network_report.xlsx'
        )
    except Exception as e:
        print("Error generating Excel export:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Initialize data and train model before starting flask server
load_data()

if __name__ == '__main__':
    # Determine port
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)