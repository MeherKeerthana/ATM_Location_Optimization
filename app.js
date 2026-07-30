let map;
let heatmapLayer = null;
let markersLayer = L.layerGroup();
let coverageCirclesLayer = L.layerGroup();

let dataset = {
    demographics: [],
    own_atms: [],
    competitor_atms: [],
    candidates: [],
    candidates_by_model: {},
};

// Global chart objects
let charts = {
    hourly: null,
    zoneWise: null,
    coverageLift: null,
    segmentAnalysis: null,
    riskWaterfall: null,
    riskDist: null,
    classDist: null,
    riskDrivers: null,
    shapSummary: null,
    riskRoc: null,
    riskPr: null,
    riskLearning: null,
    riskCalibration: null,
    riskCorrelation: null,
    predVsActual: null,
    featureImportance: null,
};

// Application state
let activeModel = "XGBoost"; // always kept in sync with the backend's best_model_name -- see fetchAnalytics
let analyticsData = null;

// Display labels for prediction models. Keys must match ATMMLModel.models
// keys on the backend (ml_engine.py) so lookups here stay in sync with
// whatever models the backend actually trains and compares.
const MODEL_DISPLAY_NAMES = {
    XGBoost: "XGBoost Regressor",
};
window.lastSelectedIds = [];
window.lastOptimizeResult = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    if (!checkLibraries()) return;

    // 1. Initialize Icons
    lucide.createIcons();

    // 2. Initialize Map
    initMap();

    // 3. Set up Slider bubble display
    setupSliders();

    // 4. Fetch initial API data (sequential call to prevent race conditions on KPI metrics)
    fetchData().then(() => {
        fetchAnalytics();
    });

    // Setup Tab Switchers (Optimization vs Risk)
    const btnOpt = document.getElementById("btn-tab-optimization");
    const btnRisk = document.getElementById("btn-tab-risk");
    const viewOpt = document.getElementById("view-optimization");
    const viewRisk = document.getElementById("view-risk");

    if (btnOpt && btnRisk && viewOpt && viewRisk) {
        btnOpt.addEventListener("click", () => {
            btnOpt.style.background = "var(--accent-blue)";
            btnOpt.style.color = "#fff";
            btnRisk.style.background = "transparent";
            btnRisk.style.color = "var(--text-secondary)";

            viewOpt.style.display = "contents";
            const insightsSec = document.querySelector(
                ".insights-section-redesigned",
            );
            if (insightsSec) insightsSec.style.display = "flex";
            viewRisk.style.display = "none";
        });

        btnRisk.addEventListener("click", () => {
            btnRisk.style.background = "var(--accent-blue)";
            btnRisk.style.color = "#fff";
            btnOpt.style.background = "transparent";
            btnOpt.style.color = "var(--text-secondary)";

            viewOpt.style.display = "none";
            const insightsSec = document.querySelector(
                ".insights-section-redesigned",
            );
            if (insightsSec) insightsSec.style.display = "none";
            viewRisk.style.display = "block";

            // Trigger risk analytics load once the risk view is visible,
            // so charts can compute layout correctly.
            window.requestAnimationFrame(() => {
                fetchRiskAnalytics();
            });
        });
    }

    const btnRiskBusiness = document.getElementById("risk-view-business");
    const btnRiskTechnical = document.getElementById("risk-view-technical");
    const businessPanel = document.getElementById("risk-view-business-panel");
    const technicalPanel = document.getElementById("risk-view-technical-panel");

    const resizeTechnicalCharts = () => {
        [
            charts.riskRoc,
            charts.riskPr,
            charts.riskLearning,
            charts.riskCalibration,
            charts.riskCorrelation,
            charts.shapSummary,
            charts.predVsActual,
            charts.featureImportance,
        ].forEach((chart) => {
            if (chart && typeof chart.resize === "function") {
                chart.resize();
            }
        });
    };

    btnRiskBusiness?.addEventListener("click", () => {
        btnRiskBusiness.classList.add("active");
        btnRiskTechnical?.classList.remove("active");
        if (businessPanel) businessPanel.style.display = "block";
        if (technicalPanel) technicalPanel.style.display = "none";
    });

    btnRiskTechnical?.addEventListener("click", () => {
        btnRiskTechnical.classList.add("active");
        btnRiskBusiness?.classList.remove("active");
        if (technicalPanel) technicalPanel.style.display = "block";
        if (businessPanel) businessPanel.style.display = "none";
        resizeTechnicalCharts();
    });

    // 5. Setup Form Submit for Optimization
    const form = document.getElementById("optimize-form");
    form.addEventListener("submit", handleOptimizeSubmit);

    // 6. Setup Toggle Listeners
    document
        .getElementById("toggle-heatmap")
        .addEventListener("change", toggleLayers);
    document
        .getElementById("toggle-competitors")
        .addEventListener("change", toggleLayers);
    document
        .getElementById("toggle-circles")
        .addEventListener("change", toggleLayers);

    // Heatmap type selection listener
    document
        .getElementById("select-heatmap-type")
        .addEventListener("change", () => {
            if (document.getElementById("toggle-heatmap").checked) {
                renderHeatmap();
            }
        });

    // Model selection is fully automatic: activeModel always tracks the
    // backend's best_model_name (see fetchAnalytics), so there is no
    // manual model-selection control here.

    // 7. Setup Table Filter Listeners
    document.getElementById("candidate-search")?.addEventListener("input", () => {
        renderCandidatesTable(window.lastSelectedIds);
    });

    document.getElementById("filter-zone")?.addEventListener("change", () => {
        renderCandidatesTable(window.lastSelectedIds);
    });

    document.getElementById("filter-roi")?.addEventListener("change", () => {
        renderCandidatesTable(window.lastSelectedIds);
    });

    // 8. Setup Tab Click Listeners for Segment Analysis
    document
        .getElementById("tab-zone")
        ?.addEventListener("click", () => switchSegmentTab("zone"));
    document
        .getElementById("tab-site-type")
        ?.addEventListener("click", () => switchSegmentTab("site-type"));
    document
        .getElementById("tab-roi-tier")
        ?.addEventListener("click", () => switchSegmentTab("roi-tier"));

    // 9. Full Report Export (PDF)
    document
        .getElementById("btn-export-pdf")
        ?.addEventListener("click", () => handleExportReport("pdf"));
});

// What fraction of `pool` this value beats, given the direction that counts
// as "better" for this feature. 1.0 = best in the pool, 0.5 = middle of pack.
function percentileScore(value, pool, higherIsBetter) {
    if (!pool || pool.length <= 1) return 0.5;
    let count = 0;
    for (const v of pool) {
        if (higherIsBetter ? v <= value : v >= value) count++;
    }
    return count / pool.length;
}

// Build a data-driven, site-specific "Key Drivers" list. Optimizers
// (MCLP/p-median/Revenue) select from the strongest candidates by design, so
// fixed absolute cutoffs (e.g. "foot traffic >= 3500") end up true for
// nearly every selected site -- producing the same two canned lines for
// every ATM in the report. Instead, rank each site's features against the
// full candidate pool (not just the selected subset) and surface whichever
// 2 factors THIS site is strongest in, with the actual numbers included so
// each ATM's drivers genuinely differ.
//
// `candidatePool` should be the full candidate list (not just selected
// sites) so percentiles are meaningful -- ranking against 3-5 selected
// sites would make everyone look like the "top 100%".
function getCandidateTopReasons(cand, candidatePool) {
    if (!cand) return ["High foot traffic corridor", "Strong demographic demand coverage"];

    const pool = (candidatePool && candidatePool.length > 1) ? candidatePool : [cand];

    const DRIVERS = [
        {
            field: "foot_traffic",
            higherIsBetter: true,
            label: (v, pctText) => `High foot traffic -- ${Math.round(v).toLocaleString()} pedestrians/day (${pctText})`,
        },
        {
            field: "pop_density",
            higherIsBetter: true,
            label: (v, pctText) => `Dense catchment population -- ${Math.round(v).toLocaleString()} nearby (${pctText})`,
        },
        {
            field: "nearby_metro_footfall",
            higherIsBetter: true,
            label: (v, pctText) => `Strong metro/transit footfall -- ${Math.round(v).toLocaleString()}/day (${pctText})`,
        },
        {
            field: "commercial_activity",
            higherIsBetter: true,
            label: (v, pctText) => `High commercial & retail activity index (${v.toFixed(1)}/100, ${pctText})`,
        },
        {
            // 0-100 score where HIGHER = closer to a mall/market hub (see
            // data_generator.py mall_score) -- previously compared against a
            // 0.4 threshold as if this were a distance, which never fired.
            field: "market_mall_proximity",
            higherIsBetter: true,
            label: (v, pctText) => `Close to a major mall/market hub (proximity score ${v.toFixed(0)}/100, ${pctText})`,
        },
        {
            field: "dist_to_nearest_competitor",
            higherIsBetter: true, // farther from a competitor = less cannibalization
            label: (v, pctText) => `Low competitor density -- nearest competitor ${v.toFixed(1)} km away (${pctText})`,
        },
        {
            field: "roi_index",
            higherIsBetter: true,
            label: (v, pctText) => `Strong ROI index of ${v.toFixed(2)}x revenue-to-rent (${pctText})`,
        },
    ];

    const scored = DRIVERS
        .filter((d) => cand[d.field] !== undefined && cand[d.field] !== null)
        .map((d) => {
            const value = Number(cand[d.field]);
            const poolValues = pool
                .map((p) => Number(p[d.field]))
                .filter((v) => !Number.isNaN(v));
            const score = percentileScore(value, poolValues, d.higherIsBetter);
            const pctRank = Math.round(score * 100);
            const pctText = pctRank >= 99 ? "top of the candidate pool" : `top ${100 - pctRank}% of candidates`;
            return { score, text: d.label(value, pctText) };
        })
        .sort((a, b) => b.score - a.score);

    if (scored.length === 0) {
        return ["Favorable ROI profile and quick payback", "Fills demographic coverage gap in zone"];
    }

    return scored.slice(0, 2).map((s) => s.text);
}

// Build a snapshot of the currently visible dashboard state and request a
// PDF or Excel report from the backend, then trigger a browser download.
async function handleExportReport(format) {
    const btn = document.getElementById("btn-export-pdf");
    const originalText = btn?.innerHTML;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="btn-icon animate-spin"></i> Preparing...`;
        lucide.createIcons();
    }

    try {
        const bestName = analyticsData?.model_metrics?.best_model_name;
        const bestMetrics =
            bestName && analyticsData?.model_metrics?.metrics
                ? analyticsData.model_metrics.metrics[bestName]
                : null;

        const payload = {
            kpis: {
                active_atms: document.getElementById("val-active-atms")?.textContent,
                coverage_pct: document.getElementById("val-coverage-pct")?.textContent,
                total_daily_tx: document.getElementById("val-total-tx")?.textContent,
                avg_uptime: document.getElementById("val-avg-uptime")?.textContent,
            },
            optimization: window.lastOptimizeResult
                ? {
                    method: window.lastOptimizeResult.method,
                    k: window.lastOptimizeResult.k,
                    radius: window.lastOptimizeResult.radius,
                    objective: window.lastOptimizeResult.objective,
                    summary: window.lastOptimizeResult.summary,
                    metrics: window.lastOptimizeResult.metrics,
                    ml_prefilter: window.lastOptimizeResult.ml_prefilter || null,
                    selected_candidates: (() => {
                        const fullCandidatePool =
                            (dataset &&
                                dataset.candidates_by_model &&
                                dataset.candidates_by_model[activeModel]) ||
                            (dataset && dataset.candidates) ||
                            [];
                        return (window.lastOptimizeResult.selected_candidates || []).map(cand => {
                            return {
                                ...cand,
                                top_reasons: getCandidateTopReasons(cand, fullCandidatePool),
                            };
                        });
                    })(),
                }
                : null,
            model_metrics: bestMetrics
                ? {
                    best_model_name: bestName,
                    r2: bestMetrics.r2_score,
                    mae: bestMetrics.mae,
                    rmse: bestMetrics.rmse,
                    accuracy: bestMetrics.accuracy,
                }
                : null,
            risk_summary: riskData
                ? {
                    risk_distribution: riskData.risk_distribution,
                    top_10_zones: riskData.top_10_zones,
                }
                : null,
        };

        const response = await fetch("/api/export/pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error("Report generation failed");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "optiatm_network_report.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Export error:", error);
        alert(`Report export failed: ${error.message}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
            lucide.createIcons();
        }
    }
}

// Initialize Leaflet Map centered on Hyderabad
function initMap() {
    const hyderabadCenter = [17.4129, 78.4484];

    map = L.map("map", {
        center: hyderabadCenter,
        zoom: 12,
        zoomControl: true,
        attributionControl: true,
    });

    // CartoDB Positron Light Tiles for a clean Light Mode theme
    L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {
            attribution:
                '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 20,
        },
    ).addTo(map);

    markersLayer.addTo(map);
    coverageCirclesLayer.addTo(map);
}

// Setup range sliders interactive values
function setupSliders() {
    const kInput = document.getElementById("opt-k");
    const kBubble = document.getElementById("bubble-k");

    kInput.addEventListener("input", () => {
        kBubble.textContent = kInput.value;
    });

    const rInput = document.getElementById("opt-radius");
    const rBubble = document.getElementById("bubble-radius");

    rInput.addEventListener("input", () => {
        rBubble.textContent = parseFloat(rInput.value).toFixed(1) + " km";
    });

    const methodSelect = document.getElementById("opt-method");
    const radiusGroup = document.getElementById("radius-group");
    const labelRadius = document.getElementById("label-radius");
    const radiusHelp = document.getElementById("radius-help-text");
    const methodHelp = document.getElementById("method-help-text");
    const objectiveGroup = document.getElementById("objective-group");

    methodSelect.addEventListener("change", () => {
        const val = methodSelect.value;
        objectiveGroup.style.display = val === "revenue" ? "block" : "none";
        if (val === "mclp") {
            radiusGroup.style.display = "block";
            labelRadius.textContent = "Service Radius (R)";
            radiusHelp.textContent =
                "Acceptable distance for a customer to walk/drive to an ATM.";
            methodHelp.textContent =
                "MCLP maximizes covered population within the target service radius.";
            rInput.min = "0.3";
            rInput.max = "2.5";
            rInput.step = "0.1";
            rInput.value = "1.0";
            rBubble.textContent = "1.0 km";
        } else if (val === "revenue") {
            radiusGroup.style.display = "block";
            labelRadius.textContent = "Min Separation Distance";
            radiusHelp.textContent =
                "Minimum distance allowed between any two selected ATMs to prevent cannibalization.";
            methodHelp.textContent =
                "ML-Revenue Maximizer selects candidates according to the objective chosen below, subject to separation.";
            rInput.min = "0.2";
            rInput.max = "1.5";
            rInput.step = "0.1";
            rInput.value = "0.5";
            rBubble.textContent = "0.5 km";
        } else if (val === "p-median") {
            radiusGroup.style.display = "none";
            methodHelp.textContent =
                "p-Median selects candidate ATMs to minimize overall average travel/walking distance for the population.";
        }
    });
}

// Format number in Indian Rupee format
function formatINR(value, isMonthly = false) {
    const formatted = "₹" + Math.round(value).toLocaleString("en-IN");
    return isMonthly ? formatted + "/mo" : formatted;
}

// Fetch Initial Spatial Data
async function fetchData() {
    try {
        const response = await fetch("/api/data");
        if (!response.ok) {
            let errMsg = "Failed to fetch spatial data";
            try {
                const errData = await response.json();
                if (errData && errData.error) errMsg = errData.error;
            } catch (e) { }
            throw new Error(errMsg);
        }

        dataset = await response.json();

        // Render initial map layers and table
        renderHeatmap();
        renderMarkers();
        renderCandidatesTable();

        // Populate filters and draw Segment Analysis Chart
        populateZoneFilter();
        switchSegmentTab("zone");

        // Update basic active ATMs count KPI
        document.getElementById("val-active-atms").textContent =
            dataset.own_atms.length;

        // Fit map bounds to Hyderabad own ATMs
        if (dataset.own_atms.length > 0) {
            const coords = dataset.own_atms.map((atm) => [
                atm.latitude,
                atm.longitude,
            ]);
            map.fitBounds(coords, { padding: [40, 40] });
        }
    } catch (error) {
        console.error("Error fetching data:", error);
        setKpiCardsToError();
        showErrorBanner(`API Error: ${error.message}`);
    }
}

// Fetch Analytics & ML stats
async function fetchAnalytics() {
    try {
        const response = await fetch("/api/analytics");
        if (!response.ok) {
            let errMsg = "Failed to fetch analytics";
            try {
                const errData = await response.json();
                if (errData && errData.error) errMsg = errData.error;
            } catch (e) { }
            throw new Error(errMsg);
        }

        analyticsData = await response.json();

        // Update KPIs safely
        const ownAtms = dataset ? dataset.own_atms : null;
        if (ownAtms && ownAtms.length > 0) {
            try {
                const totalTx = ownAtms.reduce(
                    (sum, atm) => sum + (atm.avg_daily_transactions || 0),
                    0,
                );
                const elTotalTx = document.getElementById("val-total-tx");
                if (elTotalTx) elTotalTx.textContent = totalTx.toLocaleString();

                const avgUptime =
                    ownAtms.reduce((sum, atm) => sum + (atm.uptime_pct || 0), 0) / ownAtms.length;
                const elAvgUptime = document.getElementById("val-avg-uptime");
                if (elAvgUptime) elAvgUptime.textContent = avgUptime.toFixed(2) + "%";

                const mockCoverage = calculateInitialCoverage(
                    ownAtms,
                    dataset ? dataset.demographics : [],
                );
                const elCoverage = document.getElementById("val-coverage-pct");
                if (elCoverage) elCoverage.textContent = mockCoverage.toFixed(1) + "%";
            } catch (kpiErr) {
                console.warn("KPI update warning:", kpiErr);
            }
        }

        // Update Models metrics table
        const metrics =
            analyticsData && analyticsData.model_metrics
                ? analyticsData.model_metrics.metrics
                : null;
        const bestModel =
            analyticsData && analyticsData.model_metrics
                ? analyticsData.model_metrics.best_model_name
                : null;
        updateModelsTable(metrics, bestModel);
        updateSingleModelMetrics(metrics, bestModel);

        // Always run the model the backend's R2 comparison actually picked
        // as best -- there is no manual override, so this simply stays in
        // sync with best_model_name every time analytics refreshes.
        if (bestModel && bestModel !== activeModel) {
            activeModel = bestModel;
            // fetchData() (which runs before this) renders markers/table
            // using the placeholder default model, since best_model_name
            // isn't known until this analytics response arrives. Re-render
            // now so the initial view reflects the actual best model.
            onModelChanged();
        }
        updateActiveModelBadge(activeModel);

        // Render Charts
        renderHourlyChart(analyticsData.hourly_trend);
        renderModelInsights(
            analyticsData && analyticsData.model_metrics
                ? analyticsData.model_metrics.insights
                : null,
        );
        renderZoneChart(analyticsData.zone_wise_metrics);
        renderCoverageChart(null); // Initial mock optimization state
        renderPredictedVsActualChart();
        renderFeatureImportanceChart();
    } catch (error) {
        console.error("Error fetching analytics:", error);
        setKpiCardsToError();
        showErrorBanner(`API Error: ${error.message}`);
    }
}

// Populate model comparison table with all 3 models side-by-side
function updateModelsTable(metrics, bestModel) {
    const tbody = document.getElementById("model-comparison-tbody");
    if (!tbody) return;

    if (!metrics && analyticsData && analyticsData.model_metrics) {
        metrics = analyticsData.model_metrics.metrics;
        bestModel = analyticsData.model_metrics.best_model_name;
    }

    tbody.innerHTML = "";

    if (!metrics) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:15px; color:var(--text-muted);">No model comparison metrics available.</td></tr>`;
        return;
    }

    const modelNames = Object.keys(metrics);
    modelNames.forEach((name) => {
        const m = metrics[name];
        const isBest = name === bestModel;
        const tr = document.createElement("tr");
        tr.style.cssText = `border-bottom: 1px solid var(--border-color); font-size: 0.85rem; ${isBest ? "background: rgba(37, 99, 235, 0.05);" : ""}`;

        tr.innerHTML = `
            <td style="padding: 10px; font-weight: 600;">${name}</td>
            <td style="padding: 10px; font-weight: 700; color: ${isBest ? "#16a34a" : "var(--text-primary)"};">${m.r2_score !== undefined ? Number(m.r2_score).toFixed(4) : "--"}</td>
            <td style="padding: 10px;">${m.mae !== undefined ? Number(m.mae).toFixed(2) : "--"}</td>
            <td style="padding: 10px;">${m.rmse !== undefined ? Number(m.rmse).toFixed(2) : "--"}</td>
            <td style="padding: 10px;">${m.accuracy !== undefined ? Number(m.accuracy).toFixed(2) + "%" : "--"}</td>
            <td style="padding: 10px;">
                ${isBest
                ? `<span style="background: rgba(22, 163, 74, 0.15); color: #16a34a; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">🏆 Best Model</span>`
                : `<span style="color: var(--text-muted); font-size: 0.75rem;">Baseline</span>`
            }
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Scatter plot: held-out predicted vs actual daily transactions (best model)
function renderPredictedVsActualChart() {
    const canvas = document.getElementById("chart-pred-vs-actual");
    if (!canvas || !analyticsData || !analyticsData.predicted_vs_actual) return;
    const ctx = canvas.getContext("2d");

    const points = analyticsData.predicted_vs_actual;
    const scatterData = points.map((p) => ({ x: p.actual, y: p.predicted }));

    const allVals = points.flatMap((p) => [p.actual, p.predicted]);
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    if (charts.predVsActual) charts.predVsActual.destroy();

    charts.predVsActual = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Held-out ATMs",
                    data: scatterData,
                    backgroundColor: "rgba(37, 99, 235, 0.65)",
                    borderColor: "#2563eb",
                    pointRadius: 4,
                    pointHoverRadius: 6,
                },
                {
                    label: "Perfect Prediction",
                    data: [
                        { x: minVal, y: minVal },
                        { x: maxVal, y: maxVal },
                    ],
                    type: "line",
                    borderColor: "#94a3b8",
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
                tooltip: {
                    callbacks: {
                        label: (item) =>
                            item.datasetIndex === 0
                                ? `Actual: ${item.raw.x.toLocaleString()} | Predicted: ${item.raw.y.toLocaleString()}`
                                : "",
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Actual Daily Transactions", font: { size: 10 } },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y: {
                    title: { display: true, text: "Predicted Daily Transactions", font: { size: 10 } },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
            },
        },
    });
}

// Horizontal bar chart: feature importance for the best-performing location model
function renderFeatureImportanceChart() {
    const canvas = document.getElementById("chart-feature-importance");
    if (
        !canvas ||
        !analyticsData ||
        !analyticsData.model_metrics ||
        !analyticsData.model_metrics.feature_importances
    )
        return;
    const ctx = canvas.getContext("2d");

    const cleanNames = {
        foot_traffic: "Foot Traffic",
        pop_density: "Pop Density",
        avg_income: "Avg Income",
        commercial_activity: "Retail Activity",
        dist_to_nearest_competitor: "Dist to Competitor",
        dist_to_nearest_own_atm: "Dist to Own ATM",
        nearby_metro_footfall: "Metro Footfall",
        market_mall_proximity: "Mall Proximity",
    };

    const bestModel = analyticsData.model_metrics.best_model_name;
    const importances =
        analyticsData.model_metrics.feature_importances[bestModel] || {};
    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);

    const labels = sorted.map((item) => cleanNames[item[0]] || item[0]);
    const values = sorted.map((item) => Number((item[1] * 100).toFixed(1)));

    if (charts.featureImportance) charts.featureImportance.destroy();

    charts.featureImportance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: `${bestModel} Feature Importance (%)`,
                    data: values,
                    backgroundColor: "rgba(37, 99, 235, 0.7)",
                    borderColor: "#2563eb",
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => `${item.raw}% of predictive weight`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Importance (%)", font: { size: 10 } },
                    grid: { display: false },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#0f172a", font: { size: 9 } },
                },
            },
        },
    });
}

// Fallback client side coverage calculation
function calculateInitialCoverage(atms, demographics) {
    if (!atms || !demographics || atms.length === 0 || demographics.length === 0)
        return 0;

    let coveredPop = 0;
    let totalPop = 0;
    const radius = 1.0; // km

    demographics.forEach((cell) => {
        totalPop += cell.pop_density;
        let isCovered = false;
        for (let i = 0; i < atms.length; i++) {
            const dist = getDistance(
                cell.latitude,
                cell.longitude,
                atms[i].latitude,
                atms[i].longitude,
            );
            if (dist <= radius) {
                isCovered = true;
                break;
            }
        }
        if (isCovered) {
            coveredPop += cell.pop_density;
        }
    });
    return (coveredPop / totalPop) * 100;
}

function getDistance(lat1, lon1, lat2, lon2) {
    const p = 0.017453292519943295;
    const c = Math.cos;
    const a =
        0.5 -
        c((lat2 - lat1) * p) / 2 +
        (c(lat1 * p) * c(lat2 * p) * (1 - c((lon2 - lon1) * p))) / 2;
    return 12742 * Math.asin(Math.sqrt(a));
}

// Render dynamic heatmaps
function renderHeatmap() {
    if (map) map.invalidateSize();
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }

    const heatmapType = document.getElementById("select-heatmap-type").value;
    let heatData = [];

    if (heatmapType === "demographics") {
        heatData = dataset.demographics.map((cell) => {
            const intensity =
                (cell.foot_traffic / 40000) * 0.7 + (cell.pop_density / 25000) * 0.3;
            return [cell.latitude, cell.longitude, Math.min(1.0, intensity)];
        });
    } else if (heatmapType === "own") {
        heatData = dataset.own_atms.map((atm) => {
            const intensity = atm.avg_daily_transactions / 250;
            return [atm.latitude, atm.longitude, Math.min(1.0, intensity)];
        });
    } else if (heatmapType === "competitor") {
        heatData = dataset.competitor_atms.map((comp) => {
            const intensity = comp.foot_traffic / 40000;
            return [comp.latitude, comp.longitude, Math.min(1.0, intensity)];
        });
    }

    // Leaflet.heat configurations appropriate for light map aesthetics
    heatmapLayer = L.heatLayer(heatData, {
        radius: 30,
        blur: 20,
        maxZoom: 14,
        gradient: {
            0.2: "rgba(37, 99, 235, 0.2)",
            0.4: "rgba(124, 58, 237, 0.4)",
            0.6: "rgba(234, 88, 12, 0.6)",
            0.8: "rgba(220, 38, 38, 0.7)",
            1.0: "rgba(220, 38, 38, 0.9)",
        },
    });

    if (document.getElementById("toggle-heatmap").checked) {
        heatmapLayer.addTo(map);
    }
}

// SVG pins styled for light mode
function createCustomMarker(color) {
    let colorHex = "#16a34a"; // Green
    if (color === "red") colorHex = "#dc2626";
    if (color === "orange") colorHex = "#ea580c";
    if (color === "blue") colorHex = "#2563eb";

    const svgIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="${colorHex}" stroke="#ffffff" stroke-width="2" stroke-linejoin="round">
            <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>
            <circle cx="12" cy="10" r="3.5" fill="#ffffff"/>
        </svg>
    `;

    return L.divIcon({
        html: svgIcon,
        className: "custom-map-icon",
        iconSize: [24, 24],
        iconAnchor: [12, 24],
        popupAnchor: [0, -24],
    });
}

// Render ATM and Candidate markers
function renderMarkers(selectedCandidateIds = []) {
    markersLayer.clearLayers();
    coverageCirclesLayer.clearLayers();

    const showCompetitors = document.getElementById("toggle-competitors").checked;
    const showCircles = document.getElementById("toggle-circles").checked;
    const serviceRadiusMeters =
        parseFloat(document.getElementById("opt-radius").value) * 1000;

    // 1. Own ATMs
    dataset.own_atms.forEach((atm) => {
        const marker = L.marker([atm.latitude, atm.longitude], {
            icon: createCustomMarker("green"),
        });
        marker.bindPopup(`
            <div style="font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 5px; font-weight:600; color:#16a34a;">ATM Site ${atm.atm_id}</h4>
                <p><strong>Zone:</strong> ${atm.zone_name}</p>
                <p><strong>Uptime:</strong> ${atm.uptime_pct}%</p>
                <p><strong>Daily Tx:</strong> ${atm.avg_daily_transactions} transactions</p>
                <p><strong>Foot Traffic:</strong> ${atm.foot_traffic.toLocaleString()}/day</p>
                <p><strong>Rent Cost:</strong> ${formatINR(atm.rent_cost, true)}</p>
            </div>
        `);
        markersLayer.addLayer(marker);

        if (showCircles) {
            const circle = L.circle([atm.latitude, atm.longitude], {
                radius: serviceRadiusMeters,
                color: "#16a34a",
                fillColor: "#16a34a",
                fillOpacity: 0.05,
                weight: 1,
            });
            coverageCirclesLayer.addLayer(circle);
        }
    });

    // 2. Competitors
    if (showCompetitors) {
        dataset.competitor_atms.forEach((comp) => {
            const marker = L.marker([comp.latitude, comp.longitude], {
                icon: createCustomMarker("red"),
            });
            marker.bindPopup(`
                <div style="font-family: 'Inter', sans-serif;">
                    <h4 style="margin-bottom: 5px; font-weight:600; color:#dc2626;">${comp.bank_name}</h4>
                    <p><strong>Zone:</strong> ${comp.zone_name}</p>
                    <p><strong>Foot Traffic:</strong> ${comp.foot_traffic.toLocaleString()}/day</p>
                </div>
            `);
            markersLayer.addLayer(marker);
        });
    }

    // 3. Candidates
    const activeCandidates =
        (dataset &&
            dataset.candidates_by_model &&
            dataset.candidates_by_model[activeModel]) ||
        (dataset && dataset.candidates) ||
        [];
    activeCandidates.forEach((cand) => {
        const isSelected = selectedCandidateIds.includes(cand.candidate_id);
        const color = isSelected ? "blue" : "orange";

        const marker = L.marker([cand.latitude, cand.longitude], {
            icon: createCustomMarker(color),
        });

        const statusText = isSelected
            ? `<strong style="color:#2563eb;">OPTIMIZED SELECTION</strong>`
            : `<span style="color:#ea580c;">Candidate Site</span>`;

        const paybackText =
            cand.payback_period > 0 ? `${cand.payback_period} months` : "Never";

        marker.bindPopup(`
            <div style="font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 5px; font-weight:600; color:${isSelected ? "#2563eb" : "#ea580c"};">${cand.name}</h4>
                <p><strong>Status:</strong> ${statusText}</p>
                <p><strong>Type:</strong> ${cand.site_type}</p>
                <p><strong>Rent Cost:</strong> ${formatINR(cand.rent_cost, true)}</p>
                <p><strong>Predicted Daily Tx:</strong> ${cand.predicted_daily_transactions}</p>
                <p><strong>ROI Index:</strong> ${cand.roi_index}</p>
                <p><strong>Payback Period:</strong> ${paybackText}</p>
            </div>
        `);
        markersLayer.addLayer(marker);

        if (showCircles && isSelected) {
            const circle = L.circle([cand.latitude, cand.longitude], {
                radius: serviceRadiusMeters,
                color: "#2563eb",
                fillColor: "#2563eb",
                fillOpacity: 0.08,
                weight: 1.5,
                dashArray: "4, 4",
            });
            coverageCirclesLayer.addLayer(circle);
        }
    });
}

// Render candidate list table
function renderCandidatesTable(selectedIds = window.lastSelectedIds || []) {
    const tbody = document.getElementById("candidates-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const activeCandidates =
        (dataset &&
            dataset.candidates_by_model &&
            dataset.candidates_by_model[activeModel]) ||
        (dataset && dataset.candidates) ||
        [];
    if (activeCandidates.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center">Loading candidate data...</td></tr>`;
        return;
    }

    // Get search and filter values
    const searchVal = (document.getElementById("candidate-search")?.value || "")
        .toLowerCase()
        .trim();
    const zoneVal = document.getElementById("filter-zone")?.value || "ALL";
    const roiVal = document.getElementById("filter-roi")?.value || "ALL";

    // Filter candidates
    let filtered = activeCandidates.filter((cand) => {
        // Search match: Name, ID, or Zone
        const nameMatch = cand.name
            ? cand.name.toLowerCase().includes(searchVal)
            : false;
        const idMatch = cand.candidate_id
            ? cand.candidate_id.toLowerCase().includes(searchVal)
            : false;
        const zoneMatchSearch = cand.zone_name
            ? cand.zone_name.toLowerCase().includes(searchVal)
            : false;
        const matchesSearch = !searchVal || nameMatch || idMatch || zoneMatchSearch;

        // Zone filter match
        const matchesZone = zoneVal === "ALL" || cand.zone_name === zoneVal;

        // ROI filter match
        let matchesROI = true;
        if (roiVal === "HIGH") {
            matchesROI = cand.roi_index > 1.2;
        } else if (roiVal === "MEDIUM") {
            matchesROI = cand.roi_index >= 0.8 && cand.roi_index <= 1.2;
        } else if (roiVal === "LOW") {
            matchesROI = cand.roi_index < 0.8;
        }

        return matchesSearch && matchesZone && matchesROI;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center">No candidates match the filter criteria.</td></tr>`;
        return;
    }

    // Sort by ROI index descending
    const sorted = [...filtered].sort((a, b) => b.roi_index - a.roi_index);

    sorted.forEach((cand) => {
        const isSelected = selectedIds.includes(cand.candidate_id);
        const row = document.createElement("tr");
        if (isSelected) {
            row.classList.add("selected-row");
        }

        const paybackText =
            cand.payback_period > 0 ? `${cand.payback_period} months` : "Never";
        const paybackColor =
            cand.payback_period > 0 && cand.payback_period < 18
                ? "#16a34a"
                : "#ea580c";

        row.innerHTML = `
            <td><strong>${cand.candidate_id}</strong></td>
            <td>${cand.zone_name}</td>
            <td>${cand.site_type}</td>
            <td>${formatINR(cand.rent_cost, true)}</td>
            <td>${cand.predicted_daily_transactions}</td>
            <td><span style="font-weight:600; color:${cand.roi_index > 1.2 ? "#16a34a" : "#ea580c"}">${cand.roi_index}</span></td>
            <td><span style="font-weight:600; color:${paybackColor}">${paybackText}</span></td>
            <td>
                <span class="status-pill ${isSelected ? "selected" : "candidate"}">
                    ${isSelected ? "Selected" : "Candidate"}
                </span>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Handle dynamic model dropdown changes
function onModelChanged() {
    renderMarkers(window.lastSelectedIds);
    renderCandidatesTable(window.lastSelectedIds);

    // Redraw Segment Analysis Chart
    const activeTab = document.querySelector(".tab-btn.active");
    const activeSegment = activeTab ? activeTab.id.replace("tab-", "") : "zone";
    renderSegmentAnalysisChart(activeSegment);
}

// Reflect the real active model in the "Active Prediction Model" badge --
// previously this text was hardcoded to "XGBoost Regressor (Active)" in
// index.html regardless of which model was actually selected/best.
function updateActiveModelBadge(modelName) {
    const el = document.getElementById("active-model-badge-text");
    if (!el) return;
    const label = MODEL_DISPLAY_NAMES[modelName] || modelName;
    el.textContent = `${label} (Active)`;
}

// Fill single model KPI metrics fallback
function updateSingleModelMetrics(metrics, bestModelName) {
    const modelIds = {
        XGBoost: "xgb",
    };

    for (const [name, key] of Object.entries(modelIds)) {
        const m = metrics ? metrics[name] : null;
        if (m) {
            const r2El = document.getElementById(`val-${key}-r2`);
            if (r2El) r2El.textContent = m.r2_score;

            const maeEl = document.getElementById(`val-${key}-mae`);
            if (maeEl) maeEl.textContent = m.mae;

            const rmseEl = document.getElementById(`val-${key}-rmse`);
            if (rmseEl) rmseEl.textContent = m.rmse;

            const accCell = document.getElementById(`val-${key}-accuracy`);
            if (accCell) {
                accCell.textContent = m.accuracy ? m.accuracy.toFixed(1) + "%" : "--%";
            }
        }
    }
}

// Toggle layer functions
function toggleLayers() {
    const showHeatmap = document.getElementById("toggle-heatmap").checked;
    if (showHeatmap && heatmapLayer) {
        map.addLayer(heatmapLayer);
    } else if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }
    renderMarkers(window.lastSelectedIds);
}

// Post form optimize submit
async function handleOptimizeSubmit(event) {
    event.preventDefault();

    const k = parseInt(document.getElementById("opt-k").value);
    const radius = parseFloat(document.getElementById("opt-radius").value);
    const method = document.getElementById("opt-method").value;
    const objective =
        method === "revenue"
            ? document.getElementById("opt-objective").value
            : "transactions";
    const zonesSelect = document.getElementById("opt-zones");
    const zones = zonesSelect
        ? Array.from(zonesSelect.selectedOptions).map((o) => o.value)
        : [];

    const btn = document.getElementById("btn-optimize");
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader-2" class="btn-icon animate-spin"></i> Running Optimizer...`;
    lucide.createIcons();

    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                k,
                radius,
                method,
                model_name: activeModel,
                objective,
                zones,
            }),
        });

        if (!response.ok) {
            let errMsg = "Optimization request failed";
            try {
                const errData = await response.json();
                if (errData && errData.error) errMsg = errData.error;
            } catch (e) { }
            throw new Error(errMsg);
        }

        const data = await response.json();
        const selectedIds = data.selected_candidates.map((c) => c.candidate_id);
        window.lastSelectedIds = selectedIds;
        window.lastOptimizeResult = { ...data, method, k, radius };

        // Refresh markers & table listing
        renderMarkers(selectedIds);
        renderCandidatesTable(selectedIds);

        // Show status details card
        document.getElementById("opt-status-card").style.display = "flex";
        document.getElementById("opt-status-text").textContent = data.summary;

        const coverageLift = data.metrics.coverage_increase;
        document.getElementById("lift-coverage").textContent =
            `+${coverageLift.toFixed(1)}%`;
        document.getElementById("lift-coverage").className =
            `impact-val ${coverageLift > 0 ? "text-green" : "text-muted"}`;

        const distReduction = data.metrics.avg_dist_reduction_km * 1000;
        document.getElementById("lift-dist").textContent =
            `-${distReduction.toFixed(0)}m`;
        document.getElementById("lift-dist").className =
            `impact-val ${distReduction > 0 ? "text-blue" : "text-muted"}`;

        // Update general coverage KPI card
        document.getElementById("val-coverage-pct").textContent =
            data.metrics.coverage_after.toFixed(1) + "%";

        // Re-render coverage lift chart
        renderCoverageChart(data.metrics);

        // Render Actionable Recommendations summary card stats
        const selectedCandidates = data.selected_candidates || [];
        if (selectedCandidates.length > 0) {
            document.getElementById("rec-val-coverage").textContent =
                `+${coverageLift.toFixed(1)}%`;

            const totalRoi = selectedCandidates.reduce(
                (sum, c) => sum + (c.roi_index || 0),
                0,
            );
            const avgRoi = totalRoi / selectedCandidates.length;
            document.getElementById("rec-val-roi").textContent =
                `${avgRoi.toFixed(2)}x`;

            const validPaybacks = selectedCandidates
                .filter((c) => c.payback_period > 0)
                .map((c) => c.payback_period);
            if (validPaybacks.length > 0) {
                const avgPayback =
                    validPaybacks.reduce((sum, v) => sum + v, 0) / validPaybacks.length;
                document.getElementById("rec-val-payback").textContent =
                    `${avgPayback.toFixed(1)} mo`;
            } else {
                document.getElementById("rec-val-payback").textContent = "Never";
            }

            // Generate Strategic Advisory Takeaway
            const advisoryEl = document.getElementById("rec-advisory-text");
            if (advisoryEl) {
                if (method === "mclp") {
                    advisoryEl.innerHTML = `<strong>Coverage Footprint Maximization Strategy:</strong> Placed using MCLP with a ${radius}km range. This deployment setup maximizes customer accessibility, capturing an additional ${coverageLift.toFixed(1)}% estimated population coverage in under-served sectors. Excellent for retail footprint expansion.`;
                } else if (method === "p-median") {
                    advisoryEl.innerHTML = `<strong>Travel Distance Optimization Strategy:</strong> Placed using p-Median. This configuration positions ATMs closest to the center of residential grids. By reducing customer travel friction by an average of ${distReduction.toFixed(0)}m, it increases service convenience and customer retention.`;
                } else if (method === "revenue") {
                    const objectiveCopy = {
                        transactions: {
                            label: "Transaction Volume Maximization",
                            detail:
                                "Prioritizes the highest-footfall candidate sites (e.g. metro exits and commercial plazas) regardless of rent. Recommended for maximizing market share / total transaction throughput, even if some sites carry a longer payback period.",
                        },
                        net_profit: {
                            label: "Net Profit Maximization",
                            detail:
                                "Prioritizes candidates with the largest absolute monthly profit (revenue minus rent). Recommended when the goal is maximizing total rupees earned across the portfolio.",
                        },
                        roi: {
                            label: "ROI Efficiency Maximization",
                            detail:
                                "Prioritizes candidates with the best revenue-to-rent ratio, favoring cheaper, more efficient sites even if their absolute transaction volume is lower. Recommended for capital-efficient expansion.",
                        },
                    };
                    const chosen = objectiveCopy[objective] || objectiveCopy.transactions;
                    advisoryEl.innerHTML = `<strong>Interchange Revenue Optimization Strategy — ${chosen.label}:</strong> Placed using ML-Revenue Maximizer with a ${radius}km spacing constraint to prevent self-cannibalization. ${chosen.detail}`;
                }
            }

            document.getElementById("recommendation-card-container").style.display =
                "block";
        } else {
            document.getElementById("recommendation-card-container").style.display =
                "none";
        }

        if (data.selected_candidates.length > 0) {
            const coords = data.selected_candidates.map((c) => [
                c.latitude,
                c.longitude,
            ]);
            map.panTo(coords[0]);
        }
    } catch (error) {
        console.error("Optimization error:", error);
        showErrorBanner(`Optimization Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
        lucide.createIcons();
    }
}

// Chart 1: Hourly usage Peak patterns
function renderHourlyChart(hourlyTrend) {
    const ctx = document.getElementById("chart-hourly").getContext("2d");
    const labels = Array.from(
        { length: 24 },
        (_, i) => `${String(i).padStart(2, "0")}:00`,
    );
    const values = Array.from({ length: 24 }, (_, i) => hourlyTrend[i] || 0);

    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, "rgba(37, 99, 235, 0.4)");
    gradient.addColorStop(1, "rgba(37, 99, 235, 0.02)");

    if (charts.hourly) charts.hourly.destroy();

    charts.hourly = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Hourly Volume",
                    data: values,
                    borderColor: "#2563eb",
                    borderWidth: 2,
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: "rgba(0, 0, 0, 0.05)" },
                    ticks: { color: "#475569", maxTicksLimit: 8, font: { size: 9 } },
                },
                y: {
                    grid: { color: "rgba(0, 0, 0, 0.05)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
            },
        },
    });
}

// "Model Insights & Market Takeaways" -- populated from the model's real
// feature importances/PDP results (returned by /api/analytics), not from
// hardcoded copy, so the claims here can never drift out of sync with what
// the model actually found.
function renderModelInsights(insights) {
    const listEl = document.getElementById("model-insights-list");
    if (!listEl) return;

    if (!insights || insights.length === 0) {
        listEl.innerHTML =
            '<li style="color: var(--text-secondary);">No model insights available yet.</li>';
        return;
    }

    listEl.innerHTML = insights
        .map(
            (item) => `
      <li style="display: flex; align-items: flex-start; gap: 6px">
        <span style="color: var(--accent-orange); font-weight: bold; margin-top: -1px;">▪</span>
        <span><strong>${item.title}:</strong> ${item.text}</span>
      </li>
    `,
        )
        .join("");
}



// Chart 5: Zone-wise bar chart
function renderZoneChart(zonesData) {
    const canvas = document.getElementById("chart-zone-wise");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Sort by transactions descending and take top 12 zones for layout aesthetic
    const sorted = [...zonesData]
        .sort((a, b) => b.avg_transactions - a.avg_transactions)
        .slice(0, 12);

    const labels = sorted.map((z) => z.zone);
    const transactions = sorted.map((z) => z.avg_transactions);
    const rents = sorted.map((z) => z.avg_rent / 1000); // scaled to k INR
    const rois = sorted.map((z) => z.avg_roi);

    if (charts.zoneWise) charts.zoneWise.destroy();

    charts.zoneWise = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Avg Transactions",
                    data: transactions,
                    backgroundColor: "rgba(37, 99, 235, 0.7)",
                    borderColor: "#2563eb",
                    borderWidth: 1,
                    yAxisID: "y",
                },
                {
                    label: "Rent Cost (k INR)",
                    data: rents,
                    backgroundColor: "rgba(234, 88, 12, 0.7)",
                    borderColor: "#ea580c",
                    borderWidth: 1,
                    yAxisID: "y",
                },
                {
                    label: "ROI Index",
                    data: rois,
                    borderColor: "#16a34a",
                    borderWidth: 2,
                    type: "line",
                    fill: false,
                    pointRadius: 3,
                    yAxisID: "y1",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: "#475569",
                        font: { size: 8 },
                        maxRotation: 45,
                        minRotation: 45,
                    },
                },
                y: {
                    type: "linear",
                    display: true,
                    position: "left",
                    grid: { color: "rgba(0, 0, 0, 0.05)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y1: {
                    type: "linear",
                    display: true,
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#16a34a", font: { size: 9 } },
                },
            },
        },
    });
}

// Chart 6: Coverage Before/After bar chart
function renderCoverageChart(optimizeMetrics) {
    const canvas = document.getElementById("chart-coverage-lift");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Fallback default mock data if no run has occurred yet
    const beforeCov = optimizeMetrics ? optimizeMetrics.coverage_before : 45.0;
    const afterCov = optimizeMetrics ? optimizeMetrics.coverage_after : 45.0;
    const beforeDist = optimizeMetrics
        ? optimizeMetrics.avg_dist_before_km
        : 1.25;
    const afterDist = optimizeMetrics ? optimizeMetrics.avg_dist_after_km : 1.25;

    if (charts.coverageLift) charts.coverageLift.destroy();

    charts.coverageLift = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Coverage %", "Avg Travel Dist (km)"],
            datasets: [
                {
                    label: "Before Optimization",
                    data: [beforeCov, beforeDist],
                    backgroundColor: "rgba(148, 163, 184, 0.8)",
                    borderRadius: 4,
                },
                {
                    label: "After Optimization",
                    data: [afterCov, afterDist],
                    backgroundColor: "rgba(37, 99, 235, 0.85)",
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#475569", font: { size: 10 } },
                },
                y: {
                    grid: { color: "rgba(0, 0, 0, 0.05)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
            },
        },
    });
}

// Pre-flight checks
function checkLibraries() {
    const missing = [];
    if (typeof L === "undefined") missing.push("Leaflet");
    if (typeof Chart === "undefined") missing.push("Chart.js");
    if (typeof lucide === "undefined") missing.push("Lucide Icons");

    if (missing.length > 0) {
        const errMsg = `Critical Error: Missing dependencies: ${missing.join(", ")}. Please verify that static assets successfully downloaded.`;
        console.error(errMsg);
        showErrorBanner(errMsg);
        return false;
    }
    return true;
}

// Error notification banner
function showErrorBanner(message) {
    let banner = document.getElementById("error-banner");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "error-banner";
        banner.style.cssText =
            'background: rgba(220, 38, 38, 0.95); color: #fff; padding: 16px; font-size: 15px; font-weight: 600; text-align: center; position: fixed; top: 0; left: 0; right: 0; z-index: 10000; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); display: flex; align-items: center; justify-content: center; gap: 12px; backdrop-filter: blur(4px); transition: all 0.3s ease; font-family: "Inter", sans-serif;';

        const warningIcon = document.createElement("span");
        warningIcon.innerHTML = "⚠️";

        const textSpan = document.createElement("span");
        textSpan.id = "error-banner-text";
        textSpan.textContent = message;

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "✕";
        closeBtn.style.cssText =
            "background: none; border: none; color: white; font-weight: bold; margin-left: 20px; cursor: pointer; font-size: 16px; opacity: 0.8;";
        closeBtn.onclick = () => banner.remove();

        banner.appendChild(warningIcon);
        banner.appendChild(textSpan);
        banner.appendChild(closeBtn);

        document.body.prepend(banner);
    } else {
        const textSpan = document.getElementById("error-banner-text");
        if (textSpan) textSpan.textContent = message;
    }
}

// KPI fallback Error
function setKpiCardsToError() {
    const ids = [
        "val-active-atms",
        "val-coverage-pct",
        "val-total-tx",
        "val-avg-uptime",
    ];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.textContent = "ERROR";
    });
}

// Populate zone options filter dynamically based on candidates data
function populateZoneFilter() {
    const select = document.getElementById("filter-zone");
    const optZonesSelect = document.getElementById("opt-zones");

    const zones = new Set();
    const activeCandidates =
        (dataset &&
            dataset.candidates_by_model &&
            dataset.candidates_by_model[activeModel]) ||
        (dataset && dataset.candidates) ||
        [];
    activeCandidates.forEach((cand) => {
        if (cand.zone_name) zones.add(cand.zone_name);
    });
    const sortedZones = Array.from(zones).sort();

    if (select) {
        // Clear options and insert default
        select.innerHTML = '<option value="ALL">All Zones</option>';
        sortedZones.forEach((zone) => {
            const opt = document.createElement("option");
            opt.value = zone;
            opt.textContent = zone;
            select.appendChild(opt);
        });
    }

    if (optZonesSelect) {
        // Preserve whatever the user already had selected across data refreshes
        const previouslySelected = new Set(
            Array.from(optZonesSelect.selectedOptions).map((o) => o.value),
        );
        optZonesSelect.innerHTML = "";
        sortedZones.forEach((zone) => {
            const opt = document.createElement("option");
            opt.value = zone;
            opt.textContent = zone;
            if (previouslySelected.has(zone)) opt.selected = true;
            optZonesSelect.appendChild(opt);
        });
    }
}

function getSegmentSeries(segmentType = "zone", candidates = []) {
    const buckets = {};

    candidates.forEach((c) => {
        let key = "Unknown";
        if (segmentType === "zone") {
            key = c.zone_name || "Unknown";
        } else if (segmentType === "site-type") {
            key = c.site_type || "Unknown";
        } else if (segmentType === "roi-tier") {
            if (c.roi_index > 1.2) {
                key = "High ROI";
            } else if (c.roi_index >= 0.8) {
                key = "Medium ROI";
            } else {
                key = "Low ROI";
            }
        }

        if (!buckets[key]) {
            buckets[key] = { count: 0, sumTx: 0 };
        }

        buckets[key].count += 1;
        buckets[key].sumTx += Number(c.predicted_daily_transactions || 0);
    });

    return Object.keys(buckets)
        .sort()
        .map((label) => ({
            label,
            count: buckets[label].count,
            avgTx: Math.round(buckets[label].sumTx / buckets[label].count),
        }));
}

function updateSegmentInsights(segmentType = "zone") {
    const insightsEl = document.getElementById("segment-insights-text");
    if (!insightsEl) return;

    const activeCandidates =
        (dataset &&
            dataset.candidates_by_model &&
            dataset.candidates_by_model[activeModel]) ||
        (dataset && dataset.candidates) ||
        [];

    if (activeCandidates.length === 0) {
        insightsEl.innerHTML =
            '<p style="margin:0; color:var(--text-secondary)">No segment data available.</p>';
        return;
    }

    const series = getSegmentSeries(segmentType, activeCandidates);
    if (series.length === 0) {
        insightsEl.innerHTML =
            '<p style="margin:0; color:var(--text-secondary)">No segment data available.</p>';
        return;
    }

    const sortedByAvg = [...series].sort((a, b) => b.avgTx - a.avgTx);
    const overallAvg = Math.round(
        series.reduce((sum, item) => sum + item.avgTx, 0) / series.length,
    );
    const top = sortedByAvg[0];
    const second = sortedByAvg[1] || sortedByAvg[0];
    const bottom = [...sortedByAvg].reverse()[0];
    const largest = [...series].sort((a, b) => b.count - a.count)[0];

    let bullets = [];
    if (segmentType === "roi-tier") {
        bullets = [
            `<li>📊 <strong>Largest tier:</strong> ${largest.label} accounts for ${largest.count} of the current candidates, which is the highest share in the displayed chart.</li>`,
            `<li>🏆 <strong>Highest average:</strong> ${top.label} shows the strongest average predicted volume at ${top.avgTx} transactions/day.</li>`,
            `<li>⚖️ <strong>Spread:</strong> ${bottom.label} trails the top tier by ${top.avgTx - bottom.avgTx} transactions/day on average, showing the range across the chart.</li>`,
        ];
    } else {
        bullets = [
            `<li>🏆 <strong>Top category:</strong> ${top.label} leads the current view with an average of ${top.avgTx} predicted transactions/day.</li>`,
            `<li>📈 <strong>Runner-up:</strong> ${second.label} follows closely at ${second.avgTx} transactions/day, above the overall average of ${overallAvg}.</li>`,
            `<li>⚠️ <strong>Lower-performing band:</strong> ${bottom.label} sits below the current average, indicating the weakest segment in this chart.</li>`,
        ];
    }

    insightsEl.innerHTML = `
    <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
      ${bullets.join("")}
    </ul>
  `;
}

// Render Tabbed Segment Analysis Chart
function renderSegmentAnalysisChart(segmentType = "zone") {
    const canvas = document.getElementById("chart-segment-analysis");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const activeCandidates =
        (dataset &&
            dataset.candidates_by_model &&
            dataset.candidates_by_model[activeModel]) ||
        (dataset && dataset.candidates) ||
        [];
    if (activeCandidates.length === 0) {
        updateSegmentInsights(segmentType);
        return;
    }

    updateSegmentInsights(segmentType);

    if (charts.segmentAnalysis) {
        charts.segmentAnalysis.destroy();
    }

    if (segmentType === "zone") {
        const series = getSegmentSeries("zone", activeCandidates);
        const labels = series.map((item) => item.label);
        const counts = series.map((item) => item.count);
        const avgTx = series.map((item) => item.avgTx);

        charts.segmentAnalysis = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Candidate Sites Count",
                        data: counts,
                        backgroundColor: "rgba(37, 99, 235, 0.65)",
                        borderColor: "#2563eb",
                        borderWidth: 1,
                        yAxisID: "y",
                        borderRadius: 4,
                    },
                    {
                        label: "Avg Predicted Transactions/Day",
                        data: avgTx,
                        borderColor: "#16a34a",
                        backgroundColor: "rgba(22, 163, 74, 0.1)",
                        borderWidth: 2,
                        type: "line",
                        fill: true,
                        pointRadius: 4,
                        yAxisID: "y1",
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { boxWidth: 10, font: { size: 9 } },
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: "#475569",
                            font: { size: 8 },
                            maxRotation: 45,
                            minRotation: 45,
                        },
                    },
                    y: {
                        type: "linear",
                        display: true,
                        position: "left",
                        title: { display: true, text: "Sites Count", font: { size: 9 } },
                        grid: { color: "rgba(0, 0, 0, 0.04)" },
                        ticks: { color: "#475569", font: { size: 9 } },
                    },
                    y1: {
                        type: "linear",
                        display: true,
                        position: "right",
                        title: {
                            display: true,
                            text: "Avg Daily Volume",
                            font: { size: 9 },
                        },
                        grid: { drawOnChartArea: false },
                        ticks: { color: "#16a34a", font: { size: 9 } },
                    },
                },
            },
        });
    } else if (segmentType === "site-type") {
        const series = getSegmentSeries("site-type", activeCandidates);
        const labels = series.map((item) => item.label);
        const counts = series.map((item) => item.count);
        const avgTx = series.map((item) => item.avgTx);

        charts.segmentAnalysis = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Candidate Sites Count",
                        data: counts,
                        backgroundColor: "rgba(124, 58, 237, 0.65)",
                        borderColor: "#7c3aed",
                        borderWidth: 1,
                        yAxisID: "y",
                        borderRadius: 4,
                    },
                    {
                        label: "Avg Predicted Tx/Day",
                        data: avgTx,
                        backgroundColor: "rgba(234, 88, 12, 0.65)",
                        borderColor: "#ea580c",
                        borderWidth: 1,
                        yAxisID: "y",
                        borderRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { boxWidth: 10, font: { size: 9 } },
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#475569", font: { size: 8 } },
                    },
                    y: {
                        grid: { color: "rgba(0, 0, 0, 0.04)" },
                        ticks: { color: "#475569", font: { size: 9 } },
                    },
                },
            },
        });
    } else if (segmentType === "roi-tier") {
        const series = getSegmentSeries("roi-tier", activeCandidates);
        const labels = series.map((item) => item.label);
        const counts = series.map((item) => item.count);

        charts.segmentAnalysis = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [
                    {
                        data: counts,
                        backgroundColor: [
                            "rgba(22, 163, 74, 0.75)", // Green
                            "rgba(37, 99, 235, 0.75)", // Blue
                            "rgba(234, 88, 12, 0.75)", // Orange
                        ],
                        borderColor: ["#16a34a", "#2563eb", "#ea580c"],
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { boxWidth: 10, font: { size: 9 } },
                    },
                },
                cutout: "60%",
            },
        });
    }
}

// Global function to switch Segment analysis active tab
window.switchSegmentTab = function (segmentType) {
    const tabs = ["zone", "site-type", "roi-tier"];
    tabs.forEach((t) => {
        const btn = document.getElementById(`tab-${t}`);
        if (btn) {
            if (t === segmentType) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        }
    });

    updateSegmentInsights(segmentType);
    renderSegmentAnalysisChart(segmentType);
};

// --- RISK ANALYTICS HANDLERS & CHART.JS PLOTTERS ---

// area_type values come through as raw snake_case ("transit_hub"); zone
// and site_type are already human-formatted at the source, so this only
// needs to touch area_type labels.
function formatSegmentLabel(raw) {
    return String(raw)
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Bars backed by very few sites can show an extreme percentage (e.g. a
// single flagged ATM in a 1-site zone reads as "100% risk") that looks
// just as authoritative as a bar backed by dozens of sites. Below this
// count, bars are shown with a sample-size label and reduced opacity so
// they read as indicative rather than statistically reliable.
const SEGMENT_MIN_RELIABLE_COUNT = 5;

let riskData = null;

async function fetchRiskAnalytics() {
    try {
        const response = await fetch("/api/risk-analytics");
        if (!response.ok) throw new Error("Failed to fetch risk analytics metrics");
        riskData = await response.json();

        // Calculate average risk score
        const totalProb = riskData.own_risk_list.reduce(
            (sum, item) => sum + item.probability,
            0,
        );
        const avgRate = (totalProb / riskData.own_risk_list.length) * 100;
        document.getElementById("risk-val-rate").textContent =
            avgRate.toFixed(1) + "%";
        document.getElementById("risk-val-high").textContent =
            riskData.risk_distribution.High;

        // Total screened = own ATMs scored + candidate sites scored, read
        // live from the response rather than hardcoded, so it can't go
        // stale if the underlying counts ever change.
        const totalScreened =
            riskData.own_risk_list.length + riskData.cand_risk_list.length;
        document.getElementById("risk-val-total").textContent = totalScreened;

        // Reliability = XGBoost ROC AUC (matches the "XGBoost ROC AUC" label).
        // Accuracy and ROC AUC are different metrics and can diverge on an
        // imbalanced target, so this must read roc_auc, not accuracy.
        const xgbMetrics = riskData.metrics["XGBoost"];
        if (xgbMetrics) {
            document.getElementById("risk-val-reliability").textContent =
                xgbMetrics.roc_auc.toFixed(1) + "%";
        }

        // Populate SHAP Waterfall site picker
        populateWaterfallSitePicker();

        // Render all the Risk charts
        renderRiskDistributionCharts();
        renderRiskShapSummaryChart();

        // Register Risk Segment tabs event listeners dynamically
        document
            .getElementById("tab-risk-zone")
            ?.addEventListener("click", () => switchRiskSegmentTab("zone"));
        document
            .getElementById("tab-risk-site-type")
            ?.addEventListener("click", () => switchRiskSegmentTab("site-type"));
        document
            .getElementById("tab-risk-area-type")
            ?.addEventListener("click", () => switchRiskSegmentTab("area-type"));
        document
            .getElementById("tab-risk-tenure")
            ?.addEventListener("click", () => switchRiskSegmentTab("tenure"));

        switchRiskSegmentTab("zone");
        renderRiskValidationCurves();
        renderRiskPrecisionRecallCurve();
        renderRiskLearningCurve();
        renderRiskCalibrationCurve();
        renderRiskCorrelationHeatmap();
        renderConfusionMatrix();

        // Initial waterfall select
        const picker = document.getElementById("select-waterfall-site");
        if (picker && picker.options.length > 0) {
            fetchWaterfallData(picker.value);
        }

        // Refresh icons
        lucide.createIcons();
    } catch (e) {
        console.error(e);
        showErrorBanner(`Risk Analytics Error: ${e.message}`);
    }
}

function populateWaterfallSitePicker() {
    const select = document.getElementById("select-waterfall-site");
    if (!select) return;
    select.innerHTML = "";

    // Add Own ATMs
    const ownGroup = document.createElement("optgroup");
    ownGroup.label = "Active Own ATMs";
    riskData.own_risk_list.forEach((site) => {
        const opt = document.createElement("option");
        opt.value = site.id;
        opt.textContent = `${site.id} - ${site.zone} (${(site.probability * 100).toFixed(0)}% Risk)`;
        ownGroup.appendChild(opt);
    });
    select.appendChild(ownGroup);

    // Add Candidates
    const candGroup = document.createElement("optgroup");
    candGroup.label = "Candidate Placement Sites";
    riskData.cand_risk_list.forEach((site) => {
        const opt = document.createElement("option");
        opt.value = site.id;
        opt.textContent = `${site.id} - ${site.zone} (${(site.probability * 100).toFixed(0)}% Risk)`;
        candGroup.appendChild(opt);
    });
    select.appendChild(candGroup);

    // Setup change listener
    select.onchange = (e) => {
        fetchWaterfallData(e.target.value);
    };
}

async function fetchWaterfallData(siteId) {
    try {
        const res = await fetch(`/api/risk-analytics/waterfall?site_id=${siteId}`);
        if (!res.ok) throw new Error("Failed to fetch waterfall details");
        const data = await res.json();

        // Update details panel
        document.getElementById("wf-site-name").textContent = siteId;

        if (data.base_value !== undefined) {
            const baseProbEl = document.getElementById("wf-base-prob");
            if (baseProbEl) {
                baseProbEl.textContent = `~${(data.base_value * 100).toFixed(1)}%`;
            }
        }

        let p = 0;
        let tier = "Low";
        const combined = [...riskData.own_risk_list, ...riskData.cand_risk_list];
        const found = combined.find((x) => x.id === siteId);
        if (found) {
            p = found.probability;
            tier = found.risk_tier;
        }

        document.getElementById("wf-site-risk").textContent =
            (p * 100).toFixed(1) + "%";
        const tierEl = document.getElementById("wf-site-tier");
        tierEl.textContent =
            tier === "High"
                ? "Watch List (High Risk)"
                : tier === "Medium"
                    ? "Improvement Opportunity"
                    : "Low Risk";

        let bg = "rgba(22, 163, 74, 0.12)";
        let color = "var(--accent-green)";
        if (tier === "High") {
            bg = "rgba(220, 38, 38, 0.12)";
            color = "var(--accent-red)";
        } else if (tier === "Medium") {
            bg = "rgba(234, 88, 12, 0.12)";
            color = "var(--accent-orange)";
        }
        tierEl.style.background = bg;
        tierEl.style.color = color;

        // Update Primary Diagnosis (Why)
        const whyEl = document.getElementById("wf-site-why");
        if (whyEl) {
            whyEl.textContent =
                data.top_diagnosis ||
                (found ? found.top_diagnosis : "Standard baseline performance");
        }

        renderWaterfallChart(data);
    } catch (e) {
        console.error(e);
    }
}

function renderWaterfallChart(wf) {
    const canvas = document.getElementById("chart-risk-waterfall");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const base = wf.base_value;
    const contribs = wf.contributions;
    const feat_vals = wf.feature_values;

    // Sort contributions by absolute value descending
    const sorted = Object.entries(contribs).sort(
        (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
    );

    const cleanNames = {
        foot_traffic: "Foot Traffic",
        pop_density: "Pop Density",
        avg_income: "Avg Income",
        commercial_activity: "Retail Activity",
        dist_to_nearest_competitor: "Dist to Competitor",
        dist_to_nearest_own_atm: "Dist to Own ATM",
        nearby_metro_footfall: "Metro Footfall",
        market_mall_proximity: "Mall Proximity",
        uptime_pct: "Uptime %",
        rent_cost: "Rent Cost",
    };

    const labels = [];
    const floatingData = [];
    const colors = [];

    // 1. Base value bar
    labels.push("E(f(x)) Base expected probability");
    floatingData.push([0, base]);
    colors.push("rgba(148, 163, 184, 0.7)");

    let runningVal = base;

    // 2. Contributions
    sorted.forEach(([feat, contrib]) => {
        if (Math.abs(contrib) < 0.001) return;

        const rawVal = feat_vals[feat];
        const labelText = `${cleanNames[feat] || feat} = ${rawVal.toLocaleString()}`;
        labels.push(labelText);

        const nextVal = runningVal + contrib;
        floatingData.push([runningVal, nextVal]);

        colors.push(
            contrib > 0 ? "rgba(220, 38, 38, 0.75)" : "rgba(22, 163, 74, 0.75)",
        );

        runningVal = nextVal;
    });

    // 3. Final prediction bar
    labels.push("f(x) Predicted Risk Score");
    floatingData.push([0, runningVal]);
    colors.push("rgba(37, 99, 235, 0.85)");

    if (charts.riskWaterfall) charts.riskWaterfall.destroy();

    charts.riskWaterfall = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    data: floatingData,
                    backgroundColor: colors,
                    borderColor: colors.map((c) =>
                        c
                            .replace("0.75", "1.0")
                            .replace("0.85", "1.0")
                            .replace("0.7", "1.0"),
                    ),
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const val = context.raw;
                            const diff = val[1] - val[0];
                            const label =
                                diff >= 0
                                    ? `+${(diff * 100).toFixed(1)}%`
                                    : `${(diff * 100).toFixed(1)}%`;
                            return `Interval: ${(val[0] * 100).toFixed(1)}% to ${(val[1] * 100).toFixed(1)}% (${label})`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    min: Math.min(0, ...floatingData.flat()) - 0.1,
                    max: Math.max(1, ...floatingData.flat()) + 0.1,
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: {
                        callback: function (value) {
                            return (value * 100).toFixed(0) + "%";
                        },
                        color: "#475569",
                        font: { size: 9 },
                    },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#0f172a", font: { size: 8, weight: "bold" } },
                },
            },
        },
    });
}

function renderRiskDistributionCharts() {
    const distCanvas = document.getElementById("chart-risk-dist");
    if (!distCanvas) return;
    const distCtx = distCanvas.getContext("2d");
    const dist = riskData.risk_distribution;

    if (charts.riskDist) charts.riskDist.destroy();

    charts.riskDist = new Chart(distCtx, {
        type: "bar",
        data: {
            labels: ["Low Risk (<=30%)", "Medium Risk (30-70%)", "High Risk (>70%)"],
            datasets: [
                {
                    label: "ATMs count",
                    data: [dist.Low, dist.Medium, dist.High],
                    backgroundColor: [
                        "rgba(22, 163, 74, 0.7)",
                        "rgba(234, 88, 12, 0.7)",
                        "rgba(220, 38, 38, 0.7)",
                    ],
                    borderColor: ["#16a34a", "#ea580c", "#dc2626"],
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y: {
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
            },
        },
    });

    const classCanvas = document.getElementById("chart-class-dist");
    if (!classCanvas) return;
    const classCtx = classCanvas.getContext("2d");
    const target = riskData.target_distribution;

    if (charts.classDist) charts.classDist.destroy();

    charts.classDist = new Chart(classCtx, {
        type: "doughnut",
        data: {
            labels: ["Performing (Class 0)", "Watch List (Class 1)"],
            datasets: [
                {
                    data: [target.Performing, target.Underperforming],
                    backgroundColor: [
                        "rgba(22, 163, 74, 0.75)",
                        "rgba(220, 38, 38, 0.75)",
                    ],
                    borderColor: ["#16a34a", "#dc2626"],
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 8 } },
                },
            },
            cutout: "60%",
        },
    });
}

function renderRiskShapSummaryChart() {
    const cleanNames = {
        foot_traffic: "Foot Traffic",
        pop_density: "Pop Density",
        avg_income: "Avg Income",
        commercial_activity: "Retail Activity",
        dist_to_nearest_competitor: "Dist to Competitor",
        dist_to_nearest_own_atm: "Dist to Own ATM",
        nearby_metro_footfall: "Metro Footfall",
        market_mall_proximity: "Mall Proximity",
        uptime_pct: "Uptime %",
        rent_cost: "Rent Cost",
    };

    const shapCanvas = document.getElementById("chart-shap-summary");
    if (!shapCanvas) return;
    const shapCtx = shapCanvas.getContext("2d");
    const shap = riskData.shap_summary || {};
    const sortedShap = Object.entries(shap).sort((a, b) => b[1] - a[1]);

    const shapLabels = sortedShap.map((item) => cleanNames[item[0]] || item[0]);
    const shapVals = sortedShap.map((item) => item[1]);

    if (charts.shapSummary) charts.shapSummary.destroy();

    charts.shapSummary = new Chart(shapCtx, {
        type: "bar",
        data: {
            labels: shapLabels,
            datasets: [
                {
                    label: "Mean absolute SHAP value",
                    data: shapVals,
                    backgroundColor: "rgba(124, 58, 237, 0.7)",
                    borderColor: "#7c3aed",
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#475569", font: { size: 8 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#0f172a", font: { size: 8 } },
                },
            },
        },
    });
}

window.switchRiskSegmentTab = function (segmentType) {
    const tabs = ["zone", "site-type", "area-type", "tenure"];
    tabs.forEach((t) => {
        const btn = document.getElementById(`tab-risk-${t}`);
        if (btn) {
            if (t === segmentType) btn.classList.add("active");
            else btn.classList.remove("active");
        }
    });

    const canvas = document.getElementById("chart-risk-segment");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let segmentObj = {};
    let labelText = "";
    let color = "rgba(234, 88, 12, 0.7)";
    let borderColor = "#ea580c";

    if (segmentType === "zone") {
        segmentObj = riskData.segment_by_zone;
        labelText = "Zone Risk Rate";
        color = "rgba(37, 99, 235, 0.7)";
        borderColor = "#2563eb";
    } else if (segmentType === "site-type") {
        segmentObj = riskData.segment_by_site_type;
        labelText = "Site Type Risk Rate";
        color = "rgba(124, 58, 237, 0.7)";
        borderColor = "#7c3aed";
    } else if (segmentType === "area-type") {
        segmentObj = riskData.segment_by_area_type;
        labelText = "Area Type Risk Rate";
        color = "rgba(22, 163, 74, 0.7)";
        borderColor = "#16a34a";
    } else if (segmentType === "tenure") {
        segmentObj = riskData.segment_by_tenure;
        labelText = "Tenure Bracket Risk Rate";
        color = "rgba(220, 38, 38, 0.7)";
        borderColor = "#dc2626";
    }

    // Tenure is an ordinal lifecycle variable (New -> Mid -> Mature -> Legacy);
    // re-sorting it by risk rate would scramble that progression and undercut
    // the "onboarding dip" narrative below. Every other segment type is sorted
    // riskiest-first as intended.
    const sorted =
        segmentType === "tenure"
            ? Object.entries(segmentObj)
            : Object.entries(segmentObj).sort(
                (a, b) => b[1].risk_rate - a[1].risk_rate,
            );
    const labels = sorted.map((x) => {
        const displayName =
            segmentType === "area-type" ? formatSegmentLabel(x[0]) : x[0];
        return `${displayName} (n=${x[1].count})`;
    });
    const rates = sorted.map((x) => x[1].risk_rate);

    // Bars under the reliability threshold get a faded fill so a
    // small-sample extreme (e.g. one flagged site in a 1-site zone) doesn't
    // visually compete with bars backed by a real sample.
    const barColors = sorted.map((x) =>
        x[1].count < SEGMENT_MIN_RELIABLE_COUNT
            ? color.replace("0.7", "0.2")
            : color,
    );
    const barBorderColors = sorted.map((x) =>
        x[1].count < SEGMENT_MIN_RELIABLE_COUNT ? borderColor + "55" : borderColor,
    );

    if (charts.riskSegment) charts.riskSegment.destroy();

    charts.riskSegment = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: labelText,
                    data: rates,
                    backgroundColor: barColors,
                    borderColor: barBorderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: "#475569",
                        font: { size: 8 },
                        maxRotation: 45,
                        minRotation: 45,
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: "Watch List Rate (%)",
                        font: { size: 9 },
                    },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: {
                        callback: function (v) {
                            return v + "%";
                        },
                        color: "#475569",
                        font: { size: 9 },
                    },
                },
            },
        },
    });

    const textEl = document.getElementById("risk-segment-insights-text");
    if (textEl) {
        // Every bullet below is derived directly from `sorted`/`segmentObj`
        // (the same real risk-rate data driving the chart above) -- no
        // hardcoded claims that could drift out of sync with the actual numbers.
        if (segmentType === "zone") {
            const topZone = sorted[0];
            const lowZone = sorted[sorted.length - 1];
            const spread = (topZone[1].risk_rate - lowZone[1].risk_rate).toFixed(1);
            textEl.innerHTML = `
                <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
                    <li>🚨 <strong>Highest Risk Zone:</strong> <strong>${topZone[0]}</strong> has the highest average Watch List rate at <strong>${topZone[1].risk_rate.toFixed(1)}%</strong>. Review network health here.</li>
                    <li>📍 <strong>Lowest Risk Zone:</strong> <strong>${lowZone[0]}</strong> is the most stable at <strong>${lowZone[1].risk_rate.toFixed(1)}%</strong>, a spread of <strong>${spread} pts</strong> across zones.</li>
                    <li>💡 <strong>Action:</strong> Audit the rent structures and adjust pricing fee incentives in high-risk zones.</li>
                </ul>
            `;
        } else if (segmentType === "site-type") {
            const topType = sorted[0];
            const lowType = sorted[sorted.length - 1];
            textEl.innerHTML = `
                <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
                    <li>🏢 <strong>Critical Site Type:</strong> ATMs at <strong>${topType[0]}</strong> show the highest Watch List rate of <strong>${topType[1].risk_rate.toFixed(1)}%</strong>.</li>
                    <li>🛡️ <strong>Most Resilient:</strong> <strong>${lowType[0]}</strong> is the steadiest format at <strong>${lowType[1].risk_rate.toFixed(1)}%</strong> average risk.</li>
                    <li>💡 <strong>Action:</strong> Renegotiate high leases in volatile site formats to lower ROI thresholds.</li>
                </ul>
            `;
        } else if (segmentType === "area-type") {
            const topArea = sorted[0];
            const lowArea = sorted[sorted.length - 1];
            textEl.innerHTML = `
                <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
                    <li>🏘️ <strong>Area Vulnerability:</strong> <strong>${formatSegmentLabel(topArea[0])}</strong> districts are most vulnerable with a risk rate of <strong>${topArea[1].risk_rate.toFixed(1)}%</strong>.</li>
                    <li>📈 <strong>Steadiest Areas:</strong> <strong>${formatSegmentLabel(lowArea[0])}</strong> districts run lowest at <strong>${lowArea[1].risk_rate.toFixed(1)}%</strong> average Watch List risk.</li>
                    <li>💡 <strong>Action:</strong> Prioritize high-income residential sectors over dense low-income old city grids.</li>
                </ul>
            `;
        } else if (segmentType === "tenure") {
            // sorted is chronological here (New -> Legacy), so find the riskiest
            // bucket separately rather than assuming it's sorted[0].
            const tenureEntries = Object.entries(segmentObj);
            const topTenure = [...tenureEntries].sort(
                (a, b) => b[1].risk_rate - a[1].risk_rate,
            )[0];
            const newKey = tenureEntries.find((e) => e[0].startsWith("New"));
            const restEntries = tenureEntries.filter((e) => e !== newKey);
            const restAvg = restEntries.length
                ? restEntries.reduce((sum, e) => sum + e[1].risk_rate, 0) /
                restEntries.length
                : null;
            let onboardingText =
                "Not enough tenure cohorts to compare new ATMs against the rest of the fleet.";
            if (newKey && restAvg !== null) {
                const diff = (newKey[1].risk_rate - restAvg).toFixed(1);
                onboardingText =
                    newKey[1].risk_rate > restAvg
                        ? `New-tenure ATMs are running <strong>${diff} pts</strong> higher risk than the rest of the fleet on average, consistent with an onboarding ramp-up period.`
                        : `New-tenure ATMs aren't currently running higher risk than the rest of the fleet -- an onboarding dip isn't showing up in this data.`;
            }
            textEl.innerHTML = `
                <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
                    <li>⏳ <strong>Tenure Alert:</strong> ATM cohort <strong>${topTenure[0]}</strong> displays the highest Watch List rate (<strong>${topTenure[1].risk_rate.toFixed(1)}%</strong>).</li>
                    <li>🚀 <strong>Onboarding Check:</strong> ${onboardingText}</li>
                    <li>💡 <strong>Action:</strong> Provide marketing signages for new locations and optimize Watch List legacy units.</li>
                </ul>
            `;
        }
    }
};



function renderRiskValidationCurves() {
    const rocCanvas = document.getElementById("chart-risk-roc");
    if (!rocCanvas) return;
    const rocCtx = rocCanvas.getContext("2d");

    const modelColors = {
        "Random Forest": "#2563eb",
        "Gradient Boosting": "#7c3aed",
        XGBoost: "#dc2626",
        "Linear Regression": "#475569",
    };

    const datasets = Object.keys(riskData.metrics).map((name) => {
        const curve = riskData.metrics[name].roc_curve || [];
        return {
            label: name,
            data: curve.map((pt) => ({ x: pt.fpr, y: pt.tpr })),
            borderColor: modelColors[name] || "#94a3b8",
            borderWidth: name === "XGBoost" ? 2.5 : 1.5,
            fill: false,
            pointRadius: 0,
            tension: 0.1,
        };
    });

    datasets.push({
        label: "Baseline (Random Guess)",
        data: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
        ],
        borderColor: "#94a3b8",
        borderWidth: 1,
        fill: false,
        pointRadius: 0,
        tension: 0.1,
        borderDash: [6, 4],
    });

    if (charts.riskRoc) charts.riskRoc.destroy();

    charts.riskRoc = new Chart(rocCtx, {
        type: "line",
        data: {
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    position: "bottom",
                    title: {
                        display: true,
                        text: "False Positive Rate",
                        font: { size: 10 },
                    },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y: {
                    title: {
                        display: true,
                        text: "True Positive Rate",
                        font: { size: 10 },
                    },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
            },
        },
    });
}
function renderConfusionMatrix() {
    if (!riskData || !riskData.confusion_matrix) return;

    const cm = riskData.confusion_matrix;
    const tnEl = document.getElementById("val-cm-tn");
    const fpEl = document.getElementById("val-cm-fp");
    const fnEl = document.getElementById("val-cm-fn");
    const tpEl = document.getElementById("val-cm-tp");

    if (tnEl) tnEl.textContent = cm.tn != null ? cm.tn : "--";
    if (fpEl) fpEl.textContent = cm.fp != null ? cm.fp : "--";
    if (fnEl) fnEl.textContent = cm.fn != null ? cm.fn : "--";
    if (tpEl) tpEl.textContent = cm.tp != null ? cm.tp : "--";
}

function renderRiskPrecisionRecallCurve() {
    const prCanvas = document.getElementById("chart-risk-pr");
    if (!prCanvas || !riskData || !riskData.metrics) return;
    const prCtx = prCanvas.getContext("2d");

    const datasets = Object.keys(riskData.metrics).map((name) => {
        const curve = riskData.metrics[name].pr_curve || [];
        return {
            label: name,
            data: curve.map((pt) => ({ x: pt.recall, y: pt.precision })),
            borderColor: name === "XGBoost" ? "#dc2626" : "#94a3b8",
            borderWidth: name === "XGBoost" ? 2.5 : 1.5,
            fill: false,
            pointRadius: 0,
            tension: 0.1,
        };
    });

    if (charts.riskPr) charts.riskPr.destroy();

    charts.riskPr = new Chart(prCtx, {
        type: "line",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    title: { display: true, text: "Recall", font: { size: 10 } },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                    min: 0,
                    max: 1,
                },
                y: {
                    title: { display: true, text: "Precision", font: { size: 10 } },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                    ticks: { color: "#475569", font: { size: 9 } },
                    min: 0,
                    max: 1,
                },
            },
        },
    });
}

function renderRiskLearningCurve() {
    const canvas = document.getElementById("chart-risk-learning");
    if (!canvas || !riskData || !riskData.learning_curve) return;
    const ctx = canvas.getContext("2d");

    const sizes = riskData.learning_curve.sizes || [];
    const trainScores = riskData.learning_curve.train_scores || [];
    const testScores = riskData.learning_curve.test_scores || [];

    if (charts.riskLearning) charts.riskLearning.destroy();

    charts.riskLearning = new Chart(ctx, {
        type: "line",
        data: {
            labels: sizes,
            datasets: [
                {
                    label: "Train Accuracy",
                    data: trainScores,
                    borderColor: "rgba(37, 99, 235, 0.85)",
                    backgroundColor: "rgba(37, 99, 235, 0.15)",
                    fill: true,
                    tension: 0.2,
                    pointRadius: 3,
                },
                {
                    label: "Validation Accuracy",
                    data: testScores,
                    borderColor: "rgba(220, 38, 38, 0.85)",
                    backgroundColor: "rgba(220, 38, 38, 0.15)",
                    fill: true,
                    tension: 0.2,
                    pointRadius: 3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Train Set Size", font: { size: 10 } },
                    ticks: { color: "#475569", font: { size: 9 } },
                },
                y: {
                    title: { display: true, text: "Accuracy", font: { size: 10 } },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: (value) => `${(value * 100).toFixed(0)}%`,
                        color: "#475569",
                        font: { size: 9 },
                    },
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                },
            },
        },
    });
}

function renderRiskCalibrationCurve() {
    const canvas = document.getElementById("chart-risk-calibration");
    if (!canvas || !riskData || !riskData.calibration_curve) return;
    const ctx = canvas.getContext("2d");

    const probPred = riskData.calibration_curve.prob_pred || [];
    const probTrue = riskData.calibration_curve.prob_true || [];
    const data = probPred.map((value, index) => ({
        x: value,
        y: probTrue[index] || 0,
    }));

    if (charts.riskCalibration) charts.riskCalibration.destroy();

    charts.riskCalibration = new Chart(ctx, {
        type: "line",
        data: {
            datasets: [
                {
                    label: "Calibration",
                    data,
                    borderColor: "rgba(37, 99, 235, 0.85)",
                    backgroundColor: "rgba(37, 99, 235, 0.1)",
                    fill: false,
                    tension: 0.2,
                    pointRadius: 4,
                },
                {
                    label: "Perfect Calibration",
                    data: [
                        { x: 0, y: 0 },
                        { x: 1, y: 1 },
                    ],
                    borderColor: "rgba(148, 163, 184, 0.7)",
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { boxWidth: 10, font: { size: 9 } },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: "Predicted Probability",
                        font: { size: 10 },
                    },
                    ticks: { color: "#475569", font: { size: 9 } },
                    min: 0,
                    max: 1,
                },
                y: {
                    title: {
                        display: true,
                        text: "Observed Frequency",
                        font: { size: 10 },
                    },
                    ticks: { color: "#475569", font: { size: 9 } },
                    min: 0,
                    max: 1,
                    grid: { color: "rgba(0, 0, 0, 0.04)" },
                },
            },
        },
    });
}

function renderRiskCorrelationHeatmap() {
    const container = document.getElementById("risk-correlation-full-matrix");
    if (!container || !riskData || !riskData.correlation_matrix) return;
    const corr = riskData.correlation_matrix;
    const features = Object.keys(corr);
    if (!features.length) return;

    const FEATURE_LABELS = {
        foot_traffic: "Foot Traffic",
        pop_density: "Population Density",
        avg_income: "Avg Income",
        commercial_activity: "Commercial Activity",
        dist_to_nearest_competitor: "Dist to Competitor",
        dist_to_nearest_own_atm: "Dist to Own ATM",
        nearby_metro_footfall: "Metro Footfall",
        market_mall_proximity: "Mall Proximity",
        rent_cost: "Rent Cost",
        months_in_service: "Months Service",
    };

    const getLabel = (key) => FEATURE_LABELS[key] || key.replace(/_/g, " ");

    container.innerHTML = "";

    // Main wrapper for Matrix + Legend
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "display: flex; flex-direction: column; gap: 1.2rem; width: 100%;";

    // Grid container: 1 label column (110px) + N feature columns (equal fr)
    const grid = document.createElement("div");
    grid.style.cssText = `
        display: grid;
        grid-template-columns: 110px repeat(${features.length}, minmax(42px, 1fr));
        gap: 3px;
        width: 100%;
        box-sizing: border-box;
    `;

    // 1. Top-Left Corner Cell
    const cornerCell = document.createElement("div");
    cornerCell.style.cssText = "background: var(--bg-tertiary, #f1f5f9); border-radius: 4px; border: 1px solid var(--glass-border, #cbd5e1); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; color: var(--text-muted, #64748b); padding: 4px;";
    cornerCell.textContent = "Features";
    grid.appendChild(cornerCell);

    // 2. Column Headers
    features.forEach((feature) => {
        const colHeader = document.createElement("div");
        colHeader.style.cssText = "background: var(--bg-tertiary, #f1f5f9); border-radius: 4px; border: 1px solid var(--glass-border, #cbd5e1); padding: 6px 2px; text-align: center; font-size: 0.68rem; font-weight: 700; color: var(--text-primary, #1e293b); display: flex; align-items: center; justify-content: center; word-break: break-word; line-height: 1.15;";
        colHeader.textContent = getLabel(feature);
        colHeader.title = getLabel(feature);
        grid.appendChild(colHeader);
    });

    // Color mapper for Seaborn / Matplotlib 'coolwarm' / 'vlag' red-blue diverging scale
    const getSeabornDivergingColor = (val) => {
        const r = Math.max(-1, Math.min(1, val));
        if (r > 0) {
            // Positive correlation: Soft white (r=0) to Deep Red/Crimson (r=1)
            const red = Math.round(248 - 63 * r);
            const green = Math.round(250 - 222 * r);
            const blue = Math.round(252 - 224 * r);
            return `rgb(${red}, ${green}, ${blue})`;
        } else if (r < 0) {
            // Negative correlation: Soft white (r=0) to Deep Navy Blue (r=-1)
            const absR = Math.abs(r);
            const red = Math.round(248 - 218 * absR);
            const green = Math.round(250 - 186 * absR);
            const blue = Math.round(252 - 77 * absR);
            return `rgb(${red}, ${green}, ${blue})`;
        }
        return "#f8fafc";
    };

    // 3. Matrix Rows
    features.forEach((rowFeature) => {
        // Row Header Label
        const rowHeader = document.createElement("div");
        rowHeader.style.cssText = "background: var(--bg-tertiary, #f1f5f9); border-radius: 4px; border: 1px solid var(--glass-border, #cbd5e1); padding: 6px 8px; font-size: 0.70rem; font-weight: 700; color: var(--text-primary, #1e293b); display: flex; align-items: center; justify-content: flex-start; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;";
        rowHeader.textContent = getLabel(rowFeature);
        rowHeader.title = getLabel(rowFeature);
        grid.appendChild(rowHeader);

        // Grid Cells
        features.forEach((colFeature) => {
            const val = Number((corr[rowFeature] || {})[colFeature] ?? 0);
            const formattedVal = (val >= 0 ? "+" : "") + val.toFixed(2);
            const bg = getSeabornDivergingColor(val);
            const isDarkBg = Math.abs(val) > 0.45;
            const textColor = isDarkBg ? "#ffffff" : "#0f172a";

            const cell = document.createElement("div");
            cell.style.cssText = `
                background: ${bg};
                color: ${textColor};
                border-radius: 4px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                padding: 8px 1px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-size: 0.70rem;
                font-weight: 700;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
                cursor: default;
            `;
            cell.textContent = formattedVal;
            cell.title = `${getLabel(rowFeature)} vs ${getLabel(colFeature)}: ${formattedVal}`;

            cell.addEventListener("mouseenter", () => {
                cell.style.transform = "scale(1.05)";
                cell.style.zIndex = "5";
                cell.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.15)";
            });
            cell.addEventListener("mouseleave", () => {
                cell.style.transform = "scale(1)";
                cell.style.zIndex = "1";
                cell.style.boxShadow = "none";
            });

            grid.appendChild(cell);
        });
    });

    wrapper.appendChild(grid);

    // 4. Diverging Legend Bar at Bottom
    const legend = document.createElement("div");
    legend.style.cssText = "display: flex; align-items: center; justify-content: center; gap: 1rem; margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary, #475569); font-weight: 600;";
    legend.innerHTML = `
        <span>-1.0 (Strong Negative Correlation)</span>
        <div style="height: 12px; width: 220px; border-radius: 6px; background: linear-gradient(to right, rgb(30, 64, 175), rgb(248, 250, 252), rgb(185, 28, 28)); border: 1px solid var(--glass-border, #cbd5e1);"></div>
        <span>+1.0 (Strong Positive Correlation)</span>
    `;
    wrapper.appendChild(legend);

    container.appendChild(wrapper);
}