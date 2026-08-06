// ============================================================
// app/static/js/dashboard.js
// S'exécute uniquement sur dashboard.html (vérifie la présence
// de #markowitz-results avant de faire quoi que ce soit).
// ============================================================

// ============================================================
// Comparaison des méthodes (tableau + graphique)
// ============================================================

const comparisonData = {}; // { "Markowitz": {expected_return, volatility, sharpe_ratio}, ... }
let performanceChartInstance = null;

function registerComparisonResult(methodName, data) {
    comparisonData[methodName] = {
        expected_return: data.expected_return,
        volatility: data.volatility,
        sharpe_ratio: data.sharpe_ratio,
    };
    renderComparisonTable();
    renderComparisonChart();
}

function renderComparisonTable() {
    const container = document.getElementById('comparison-table');
    if (!container) return;

    const methods = Object.keys(comparisonData);
    if (methods.length === 0) {
        container.innerHTML = '';
        return;
    }

    const rows = methods.map(method => {
        const d = comparisonData[method];
        return `
            <tr>
                <td><strong>${method}</strong></td>
                <td class="text-end">${(d.expected_return * 100).toFixed(2)} %</td>
                <td class="text-end">${(d.volatility * 100).toFixed(2)} %</td>
                <td class="text-end">${d.sharpe_ratio.toFixed(4)}</td>
            </tr>
        `;
    }).join('');

    container.innerHTML = `
        <div class="card">
            <div class="card-header">Comparaison synthétique</div>
            <div class="card-body">
                <table class="table table-sm mb-0">
                    <thead>
                        <tr>
                            <th>Méthode</th>
                            <th class="text-end">Rendement attendu</th>
                            <th class="text-end">Volatilité</th>
                            <th class="text-end">Ratio de Sharpe</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
}

function renderComparisonChart() {
    const canvas = document.getElementById('performance-chart');
    if (!canvas) return;

    const methods = Object.keys(comparisonData);
    if (methods.length === 0) return;

    const returns = methods.map(m => +(comparisonData[m].expected_return * 100).toFixed(2));
    const vols = methods.map(m => +(comparisonData[m].volatility * 100).toFixed(2));
    const sharpes = methods.map(m => +comparisonData[m].sharpe_ratio.toFixed(4));

    if (performanceChartInstance) performanceChartInstance.destroy();

    performanceChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: methods,
            datasets: [
                {
                    label: 'Rendement attendu (%)',
                    data: returns,
                    backgroundColor: '#4e79a7',
                    yAxisID: 'y',
                },
                {
                    label: 'Volatilité (%)',
                    data: vols,
                    backgroundColor: '#e15759',
                    yAxisID: 'y',
                },
                {
                    label: 'Ratio de Sharpe',
                    data: sharpes,
                    backgroundColor: '#59a14f',
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'Comparaison des méthodes d\'optimisation' },
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Rendement / Volatilité (%)' },
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Ratio de Sharpe' },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const markowitzDiv = document.getElementById('markowitz-results');
    if (!markowitzDiv) return; // pas sur la page dashboard, on ne fait rien

    const tickersJson = localStorage.getItem('portfolioTickers');

    if (!tickersJson) {
        markowitzDiv.innerHTML = `
            <p class="text-muted">
            Aucune donnée disponible. Veuillez d'abord télécharger des
            données sur la page <a href="/data">Données</a>.
            </p>`;
        return;
    }

    const tickers = JSON.parse(tickersJson);

    markowitzDiv.innerHTML = '<p class="text-muted">Calcul en cours...</p>';

    // Premier calcul au chargement, Markowitz pur (sans contrainte)
    await runMarkowitz(tickers, 1.0);
    await initBlackLitterman(tickers);
    await runML(tickers);

    const backtestBtn = document.getElementById('backtest-btn');
    if (backtestBtn) {
        backtestBtn.addEventListener('click', () => runBacktest(tickers));
    }
});

// ============================================================
// Machine Learning (Hierarchical Risk Parity)
// ============================================================

async function runML(tickers) {
    const outputDiv = document.getElementById('ml-results');
    if (!outputDiv) return;

    outputDiv.innerHTML = '<p class="text-muted">Calcul en cours...</p>';

    try {
        const response = await fetch('/api/optimization/ml', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, risk_free_rate: 0.02 }),
        });
        const result = await response.json();

        if (!response.ok || result.error) {
            outputDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Erreur inconnue'}</div>`;
            return;
        }

        outputDiv.innerHTML = renderMLResult(result.data);
        registerComparisonResult('ML prédictif', result.data);
    } catch (err) {
        outputDiv.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
    }
}

function renderMLResult(data) {
    const rows = Object.entries(data.weights)
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, poids]) => `
            <tr>
                <td>${ticker}</td>
                <td class="text-end">${(poids * 100).toFixed(2)} %</td>
            </tr>
        `).join('');

    const modelNames = {
        ridge: 'Ridge Regression',
        random_forest: 'Random Forest',
        gradient_boosting: 'Gradient Boosting',
        xgboost: 'XGBoost',
        lightgbm: 'LightGBM',
        gaussian_process: 'Gaussian Process',
    };
    const modelLabel = data.ml_model_selected
        ? (modelNames[data.ml_model_selected] || data.ml_model_selected)
        : null;

    const modelBadge = modelLabel
        ? `<span class="badge bg-success">Vues générées par ${modelLabel}</span>`
        : `<span class="badge bg-secondary">Pas assez de données — équilibre de marché pur</span>`;

    const diag = data.ml_model_diagnostics || {};
    const diagRows = Object.entries(diag)
        .map(([ticker, d]) => `
            <tr>
                <td>${ticker}</td>
                <td class="text-end">${d.mae.toFixed(4)}</td>
                <td class="text-end">${d.rmse.toFixed(4)}</td>
                <td class="text-end">${d.r2.toFixed(4)}</td>
                <td class="text-end">${(d.direction_accuracy * 100).toFixed(1)} %</td>
            </tr>
        `).join('');
    const diagnosticsHtml = (modelLabel && diagRows)
        ? `
            <details class="mb-3">
                <summary class="small text-muted" style="cursor:pointer;">
                    Qualité du modèle par actif (hold-out chronologique)
                </summary>
                <table class="table table-sm mt-2">
                    <thead>
                        <tr>
                            <th>Actif</th><th class="text-end">MAE</th>
                            <th class="text-end">RMSE</th><th class="text-end">R²</th>
                            <th class="text-end">Direction</th>
                        </tr>
                    </thead>
                    <tbody>${diagRows}</tbody>
                </table>
            </details>
        `
        : '';

    const confidencePct = ((data.confidence_used || 0) * 100).toFixed(1);
    const confidenceHtml = `
        <p class="small text-muted mb-1">
            Confiance appliquée aux vues (auto-calibrée sur le R²) : <strong>${confidencePct} %</strong>
        </p>
    `;

    const excludedHtml = (data.views_excluded && data.views_excluded.length > 0)
        ? `
            <p class="small text-muted mb-3">
                Aucune vue soumise pour ${data.views_excluded.join(', ')} (rendement prédit négatif) —
                équilibre de marché utilisé pour ces actifs.
            </p>
        `
        : '<p class="small text-muted mb-3"></p>';

    const warningsHtml = (data.warnings || []).map(w => `
        <div class="alert alert-warning py-2 px-3 small mb-3">⚠️ ${w}</div>
    `).join('');

    const predictedRows = Object.entries(data.ml_predicted_returns || {})
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, r]) => `
            <tr><td>${ticker}</td><td class="text-end">${(r * 100).toFixed(2)} %</td></tr>
        `).join('');

    return `
        <p class="mb-2">${modelBadge}</p>
        ${diagnosticsHtml}
        ${confidenceHtml}
        ${excludedHtml}
        ${warningsHtml}
        <table class="table table-sm mb-3">
            <thead><tr><th>Actif</th><th class="text-end">Poids</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <p class="mb-1"><strong>Rendement attendu :</strong> ${(data.expected_return * 100).toFixed(2)} %</p>
        <p class="mb-1"><strong>Volatilité :</strong> ${(data.volatility * 100).toFixed(2)} %</p>
        <p class="mb-3"><strong>Ratio de Sharpe :</strong> ${data.sharpe_ratio.toFixed(4)}</p>
        <details>
            <summary class="small text-muted" style="cursor:pointer;">Voir les rendements prédits par le modèle (vues soumises à Black-Litterman)</summary>
            <table class="table table-sm mt-2">
                <thead><tr><th>Actif</th><th class="text-end">Rendement prédit</th></tr></thead>
                <tbody>${predictedRows}</tbody>
            </table>
        </details>
    `;
}

// ============================================================
// Backtest hors-échantillon
// ============================================================

let backtestChartInstance = null;

async function runBacktest(tickers) {
    const outputDiv = document.getElementById('backtest-output');
    outputDiv.innerHTML = '<p class="text-muted">Backtest en cours (plusieurs optimisations successives, peut prendre quelques secondes)...</p>';

    try {
        const response = await fetch('/api/optimization/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, risk_free_rate: 0.02, min_train_years: 2 }),
        });
        const result = await response.json();

        if (!response.ok || result.error) {
            outputDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Erreur inconnue'}</div>`;
            return;
        }

        outputDiv.innerHTML = renderBacktestResult(result.data);
        renderBacktestChart(result.data);
    } catch (err) {
        outputDiv.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
    }
}

function renderBacktestResult(data) {
    const { markowitz, black_litterman, ml } = data;

    const summaryRow = (label, summary) => `
        <tr>
            <td><strong>${label}</strong></td>
            <td class="text-end">${(summary.avg_realized_return * 100).toFixed(2)} %</td>
            <td class="text-end">${(summary.avg_realized_volatility * 100).toFixed(2)} %</td>
            <td class="text-end">${summary.avg_realized_sharpe.toFixed(4)}</td>
            <td class="text-end">${summary.std_realized_sharpe.toFixed(4)}</td>
            <td class="text-end">${summary.avg_realized_sortino.toFixed(4)}</td>
            <td class="text-end">${(summary.avg_cagr * 100).toFixed(2)} %</td>
            <td class="text-end">${summary.avg_calmar_ratio.toFixed(4)}</td>
            <td class="text-end">${(summary.avg_max_drawdown * 100).toFixed(2)} %</td>
            <td class="text-end">${(summary.worst_max_drawdown * 100).toFixed(2)} %</td>
            <td class="text-end">${(summary.win_rate * 100).toFixed(0)} %</td>
        </tr>
    `;

    const windowRows = (windows) => windows.map(w => `
        <tr>
            <td>${w.test_year}</td>
            <td class="text-end">${(w.predicted_return * 100).toFixed(2)} %</td>
            <td class="text-end">${(w.realized_return * 100).toFixed(2)} %</td>
            <td class="text-end">${w.predicted_sharpe.toFixed(2)}</td>
            <td class="text-end">${w.realized_sharpe.toFixed(2)}</td>
        </tr>
    `).join('');

    return `
        <p class="small text-muted">
            ${markowitz.summary.n_windows} fenêtre(s) de test :
            ${markowitz.windows.map(w => w.test_year).join(', ')}
        </p>

        <h6>Résumé (moyenne sur toutes les fenêtres de test)</h6>
        <table class="table table-sm mb-4">
            <thead>
                <tr>
                    <th>Méthode</th>
                    <th class="text-end">Rendement réalisé (moy.)</th>
                    <th class="text-end">Volatilité réalisée (moy.)</th>
                    <th class="text-end">Sharpe réalisé (moy.)</th>
                    <th class="text-end">Sharpe réalisé (écart-type)</th>
                    <th class="text-end">Sortino (moy.)</th>
                    <th class="text-end">CAGR (moy.)</th>
                    <th class="text-end">Calmar (moy.)</th>
                    <th class="text-end">Max Drawdown (moy.)</th>
                    <th class="text-end">Max Drawdown (pire)</th>
                    <th class="text-end">Taux de réussite</th>
                </tr>
            </thead>
            <tbody>
                ${summaryRow('Markowitz', markowitz.summary)}
                ${summaryRow('Black-Litterman', black_litterman.summary)}
                ${summaryRow('ML (BL + IA)', ml.summary)}
            </tbody>
        </table>

        <div class="row">
            <div class="col-md-4">
                <h6>Détail par année — Markowitz</h6>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Année</th>
                            <th class="text-end">Rend. prédit</th>
                            <th class="text-end">Rend. réalisé</th>
                            <th class="text-end">Sharpe préd.</th>
                            <th class="text-end">Sharpe réal.</th>
                        </tr>
                    </thead>
                    <tbody>${windowRows(markowitz.windows)}</tbody>
                </table>
            </div>
            <div class="col-md-4">
                <h6>Détail par année — Black-Litterman</h6>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Année</th>
                            <th class="text-end">Rend. prédit</th>
                            <th class="text-end">Rend. réalisé</th>
                            <th class="text-end">Sharpe préd.</th>
                            <th class="text-end">Sharpe réal.</th>
                        </tr>
                    </thead>
                    <tbody>${windowRows(black_litterman.windows)}</tbody>
                </table>
            </div>
            <div class="col-md-4">
                <h6>Détail par année — ML (BL + IA)</h6>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Année</th>
                            <th class="text-end">Rend. prédit</th>
                            <th class="text-end">Rend. réalisé</th>
                            <th class="text-end">Sharpe préd.</th>
                            <th class="text-end">Sharpe réal.</th>
                        </tr>
                    </thead>
                    <tbody>${windowRows(ml.windows)}</tbody>
                </table>
            </div>
        </div>
        <canvas id="backtest-chart" class="mt-3"></canvas>
    `;
}

function renderBacktestChart(data) {
    const canvas = document.getElementById('backtest-chart');
    if (!canvas) return;

    const years = data.markowitz.windows.map(w => w.test_year);
    const markowitzSharpes = data.markowitz.windows.map(w => w.realized_sharpe);
    const blSharpes = data.black_litterman.windows.map(w => w.realized_sharpe);
    const mlSharpes = data.ml.windows.map(w => w.realized_sharpe);

    if (backtestChartInstance) backtestChartInstance.destroy();

    backtestChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: years,
            datasets: [
                { label: 'Markowitz — Sharpe réalisé', data: markowitzSharpes, backgroundColor: '#4e79a7' },
                { label: 'Black-Litterman — Sharpe réalisé', data: blSharpes, backgroundColor: '#f28e2b' },
                { label: 'ML (BL + IA) — Sharpe réalisé', data: mlSharpes, backgroundColor: '#59a14f' },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'Ratio de Sharpe RÉELLEMENT obtenu, année par année (hors-échantillon)' },
            },
            scales: {
                y: { title: { display: true, text: 'Ratio de Sharpe réalisé' } },
            },
        },
    });
}

async function runMarkowitz(tickers, maxWeight) {
    const outputDiv = document.getElementById('markowitz-results');
    outputDiv.innerHTML = '<p class="text-muted">Calcul en cours...</p>';

    try {
        const response = await fetch('/api/optimization/markowitz', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, risk_free_rate: 0.02, max_weight: maxWeight }),
        });
        const result = await response.json();

        if (!response.ok || result.error) {
            outputDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Erreur inconnue'}</div>`;
            return;
        }

        outputDiv.innerHTML = renderMarkowitzResult(result.data);
        registerComparisonResult('Markowitz', result.data);
    } catch (err) {
        outputDiv.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
    }
}

// ============================================================
// Black-Litterman
// ============================================================

async function initBlackLitterman(tickers) {
    const blDiv = document.getElementById('black-litterman-results');
    if (!blDiv) return;

    renderViewsForm(tickers);

    blDiv.innerHTML = '<p class="text-muted">Calcul en cours...</p>';
    await runBlackLitterman(tickers, {}, {});
}

// ------------------------------------------------------------
// Formulaire de saisie des vues (ticker + rendement attendu +
// confiance). Sans ce formulaire, l'appel à /black-litterman ne
// recevait jamais que des vues vides ({}), le portefeuille rendu
// était donc toujours le pur équilibre de marché, jamais l'ajustement
// bayésien avec vue investisseur — voir black_litterman.py, dont le
// moteur (build_views_matrices / black_litterman_posterior) attend
// justement ces vues en entrée.
// ------------------------------------------------------------

let blViewRowCount = 0;

function renderViewsForm(tickers) {
    const container = document.getElementById('bl-views-form');
    if (!container) return;

    container.innerHTML = `
        <p class="fw-bold mb-2 small">Vos vues (optionnel)</p>
        <div id="bl-views-rows"></div>
        <div class="d-flex gap-2 mb-2">
            <button type="button" id="bl-add-view-btn" class="btn btn-sm btn-outline-secondary">+ Ajouter une vue</button>
        </div>
        <div class="d-flex gap-2">
            <button type="button" id="bl-apply-views-btn" class="btn btn-sm btn-primary">Appliquer les vues</button>
            <button type="button" id="bl-reset-views-btn" class="btn btn-sm btn-outline-danger">Réinitialiser</button>
        </div>
        <hr class="mt-3 mb-0">
    `;

    document.getElementById('bl-add-view-btn').addEventListener('click', () => addViewRow(tickers));
    document.getElementById('bl-apply-views-btn').addEventListener('click', () => applyViews(tickers));
    document.getElementById('bl-reset-views-btn').addEventListener('click', () => resetViews(tickers));

    addViewRow(tickers); // une ligne vide au départ, pour inviter à la saisie
}

function addViewRow(tickers) {
    const rowsDiv = document.getElementById('bl-views-rows');
    if (!rowsDiv) return;

    const rowId = `bl-view-row-${blViewRowCount++}`;
    const options = tickers.map(t => `<option value="${t}">${t}</option>`).join('');

    const row = document.createElement('div');
    row.className = 'row g-1 mb-2 align-items-center';
    row.id = rowId;
    row.innerHTML = `
        <div class="col-4">
            <select class="form-select form-select-sm bl-view-ticker">${options}</select>
        </div>
        <div class="col-3">
            <input type="number" step="1" class="form-control form-control-sm bl-view-return" placeholder="Rend. %">
        </div>
        <div class="col-3">
            <input type="number" step="1" min="0" max="100" value="50" class="form-control form-control-sm bl-view-confidence" placeholder="Conf. %">
        </div>
        <div class="col-2">
            <button type="button" class="btn btn-sm btn-outline-danger w-100 bl-remove-view-btn" title="Retirer cette vue">×</button>
        </div>
    `;
    rowsDiv.appendChild(row);

    row.querySelector('.bl-remove-view-btn').addEventListener('click', () => row.remove());
}

function collectViewsFromForm() {
    const views = {};
    const confidences = {};

    document.querySelectorAll('#bl-views-rows > div').forEach(row => {
        const ticker = row.querySelector('.bl-view-ticker').value;
        const returnPct = row.querySelector('.bl-view-return').value;
        const confPct = row.querySelector('.bl-view-confidence').value;

        if (returnPct === '' || returnPct === null) return; // ligne vide -> ignorée, pas de vue soumise

        views[ticker] = parseFloat(returnPct) / 100;
        // Bornes cohérentes avec build_views_matrices() côté backend
        // (évite une division par 0 dans le calcul d'Omega)
        const conf = confPct === '' ? 0.5 : parseFloat(confPct) / 100;
        confidences[ticker] = Math.min(Math.max(conf, 0.0001), 0.999999);
    });

    return { views, confidences };
}

async function applyViews(tickers) {
    const { views, confidences } = collectViewsFromForm();
    await runBlackLitterman(tickers, views, confidences);
}

async function resetViews(tickers) {
    document.getElementById('bl-views-rows').innerHTML = '';
    addViewRow(tickers);
    await runBlackLitterman(tickers, {}, {});
}

async function runBlackLitterman(tickers, views = {}, confidences = {}) {
    const outputDiv = document.getElementById('black-litterman-results');
    outputDiv.innerHTML = '<p class="text-muted">Calcul en cours...</p>';

    try {
        const response = await fetch('/api/optimization/black-litterman', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tickers,
                risk_free_rate: 0.02,
                tau: 0.05,
                views,
                confidences,
            }),
        });
        const result = await response.json();

        if (!response.ok || result.error) {
            outputDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Erreur inconnue'}</div>`;
            return;
        }

        outputDiv.innerHTML = renderBlackLittermanResult(result.data, views);
        registerComparisonResult('Black-Litterman', result.data);
    } catch (err) {
        outputDiv.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
    }
}

function renderBlackLittermanResult(data, views = {}) {
    const rows = Object.entries(data.weights)
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, poids]) => `
            <tr>
                <td>${ticker}</td>
                <td class="text-end">${(poids * 100).toFixed(2)} %</td>
            </tr>
        `).join('');

    const warningsHtml = (data.warnings || []).map(w => `
        <div class="alert alert-warning py-2 px-3 small mb-3">⚠️ ${w}</div>
    `).join('');

    const hasViews = Object.keys(views).length > 0;
    const viewsBadge = hasViews
        ? `<span class="badge bg-primary mb-2">Vue(s) appliquée(s) : ${Object.keys(views).join(', ')}</span>`
        : `<span class="badge bg-secondary mb-2">Équilibre de marché pur (aucune vue)</span>`;

    // Comparaison Pi (équilibre) vs rendements postérieurs, pour
    // visualiser concrètement l'effet de la vue sur chaque actif
    // (les deux séries sont déjà renvoyées par optimize_black_litterman)
    const equilibriumRows = Object.keys(data.equilibrium_returns || {})
        .map(ticker => `
            <tr>
                <td>${ticker}</td>
                <td class="text-end">${(data.equilibrium_returns[ticker] * 100).toFixed(2)} %</td>
                <td class="text-end">${(data.posterior_returns[ticker] * 100).toFixed(2)} %</td>
            </tr>
        `).join('');

    return `
        ${viewsBadge}
        ${warningsHtml}
        <table class="table table-sm mb-3">
            <thead><tr><th>Actif</th><th class="text-end">Poids</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <p class="mb-1"><strong>Rendement attendu :</strong> ${(data.expected_return * 100).toFixed(2)} %</p>
        <p class="mb-1"><strong>Volatilité :</strong> ${(data.volatility * 100).toFixed(2)} %</p>
        <p class="mb-3"><strong>Ratio de Sharpe :</strong> ${data.sharpe_ratio.toFixed(4)}</p>
        <details>
            <summary class="small text-muted" style="cursor:pointer;">Voir l'effet des vues (équilibre vs postérieur)</summary>
            <table class="table table-sm mt-2">
                <thead><tr><th>Actif</th><th class="text-end">Pi (équilibre)</th><th class="text-end">Postérieur</th></tr></thead>
                <tbody>${equilibriumRows}</tbody>
            </table>
        </details>
    `;
}

function renderMarkowitzResult(data) {
    const rows = Object.entries(data.weights)
        .sort((a, b) => b[1] - a[1]) // tri décroissant par poids
        .map(([ticker, poids]) => `
            <tr>
                <td>${ticker}</td>
                <td class="text-end">${(poids * 100).toFixed(2)} %</td>
            </tr>
        `).join('');

    return `
        <table class="table table-sm mb-3">
            <thead><tr><th>Actif</th><th class="text-end">Poids</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <p class="mb-1"><strong>Rendement attendu :</strong> ${(data.expected_return * 100).toFixed(2)} %</p>
        <p class="mb-1"><strong>Volatilité :</strong> ${(data.volatility * 100).toFixed(2)} %</p>
        <p class="mb-0"><strong>Ratio de Sharpe :</strong> ${data.sharpe_ratio.toFixed(4)}</p>
    `;
}