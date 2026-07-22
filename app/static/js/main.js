document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('asset-selection-form');
    if (!form) return;

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const tickers = document.getElementById('tickers').value
            .split(',')
            .map(t => t.trim().toUpperCase())
            .filter(Boolean);
        const start = document.getElementById('start-date').value;
        const end = document.getElementById('end-date').value;

        const resultsDiv = document.getElementById('results');
        const submitBtn = document.getElementById('submit-btn');

        submitBtn.disabled = true;
        submitBtn.textContent = 'Téléchargement en cours...';
        resultsDiv.innerHTML = '<p class="text-muted">Récupération des données depuis Yahoo Finance...</p>';

        try {
            const response = await fetch('/api/data/fetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers, start, end }),
            });
            const result = await response.json();

            if (!response.ok || result.error) {
                resultsDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Erreur inconnue'}</div>`;
            } else {
                resultsDiv.innerHTML = renderSummary(result.data);

                // AJOUT : mémorise les tickers/dates pour que le dashboard
                // puisse relancer l'optimisation Markowitz sans redemander
                // de formulaire.
                localStorage.setItem('portfolioTickers', JSON.stringify(tickers));
                localStorage.setItem('portfolioStart', start);
                localStorage.setItem('portfolioEnd', end);

                // AJOUT : dessiner les graphiques (prix, rendements, corrélation)
                // si le backend a bien renvoyé chart_data dans la réponse.
                if (result.data.chart_data && typeof renderAllCharts === 'function') {
                    renderAllCharts(result.data.chart_data);
                } else if (!result.data.chart_data) {
                    console.warn('chart_data absent de la réponse du backend — vérifiez data_collector.py');
                }
            }
        } catch (err) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Télécharger les données';
        }
    });
});

function renderSummary(data) {
    const { tickers, start_date, end_date, nb_observations, preview, stats } = data;

    const statsRows = tickers.map(t => `
        <tr>
            <td>${t}</td>
            <td>${(stats[t].rendement_moyen_annualise * 100).toFixed(2)} %</td>
            <td>${(stats[t].volatilite_annualisee * 100).toFixed(2)} %</td>
        </tr>
    `).join('');

    // Ordre explicite des colonnes : Date en premier, puis les tickers
    // (le JSON renvoyé par Flask trie les clés par ordre alphabétique,
    // ce qui place "Date" au mauvais endroit si on ne force pas l'ordre ici)
    const columnOrder = ['Date', ...tickers];

    const previewHeaders = columnOrder.map(k => `<th>${k}</th>`).join('');
    const previewRows = preview.map(row => `
        <tr>${columnOrder.map(k => {
            const v = row[k];
            return `<td>${typeof v === 'number' ? v.toFixed(2) : v}</td>`;
        }).join('')}</tr>
    `).join('');

    return `
        <div class="alert alert-success">
            Données téléchargées avec succès : <strong>${nb_observations}</strong> jours de cotation
            du <strong>${start_date}</strong> au <strong>${end_date}</strong>.
        </div>

        <h5 class="mt-3">Statistiques annualisées</h5>
        <table class="table table-sm table-bordered">
            <thead><tr><th>Actif</th><th>Rendement moyen</th><th>Volatilité</th></tr></thead>
            <tbody>${statsRows}</tbody>
        </table>

        <h5 class="mt-3">Aperçu des derniers cours (clôture)</h5>
        <div class="table-responsive">
            <table class="table table-sm table-striped">
                <thead><tr>${previewHeaders}</tr></thead>
                <tbody>${previewRows}</tbody>
            </table>
        </div>
        <p class="text-muted small">Les fichiers complets ont été sauvegardés dans data/raw/ et data/processed/.</p>
    `;
}