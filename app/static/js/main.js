document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('asset-selection-form');
    if (!form) return;

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const tickers = document.getElementById('tickers').value
            .split(',')
            .map(t => t.trim())
            .filter(Boolean);
        const start = document.getElementById('start-date').value;
        const end = document.getElementById('end-date').value;

        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<p class="text-muted">Chargement...</p>';

        try {
            const response = await fetch('/api/data/fetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers, start, end }),
            });
            const data = await response.json();

            if (data.error) {
                resultsDiv.innerHTML = `<p class="text-danger">${data.error}</p>`;
            } else {
                resultsDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            }
        } catch (err) {
            resultsDiv.innerHTML = `<p class="text-danger">Erreur : ${err.message}</p>`;
        }
    });
});
