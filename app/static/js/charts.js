// ============================================================
// app/static/js/charts.js
// Nécessite Chart.js déjà chargé (CDN dans base.html).
// Appelez renderAllCharts(chartData) après réception du JSON
// (soit depuis la réponse de /data/fetch, soit /data/chart-data).
// ============================================================

let priceChartInstance = null;
let returnsChartInstance = null;

const CHART_COLORS = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
  "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
];

function renderAllCharts(chartData) {
  renderPriceChart(chartData.prices);
  renderReturnsChart(chartData.returns);
  renderCorrelationMatrix(chartData.correlation);
}

// --------------------- Graphique des prix ---------------------
function renderPriceChart(pricesData) {
  const ctx = document.getElementById("priceChart").getContext("2d");

  const datasets = Object.keys(pricesData.series).map((ticker, i) => ({
    label: ticker,
    data: pricesData.series[ticker],
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: "transparent",
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1,
  }));

  if (priceChartInstance) priceChartInstance.destroy();

  priceChartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: pricesData.dates, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        title: { display: true, text: "Évolution des prix ajustés" },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Prix ($)" } },
      },
    },
  });
}

// --------------------- Graphique des rendements ---------------------
function renderReturnsChart(returnsData) {
  const ctx = document.getElementById("returnsChart").getContext("2d");

  const datasets = Object.keys(returnsData.series).map((ticker, i) => ({
    label: ticker,
    data: returnsData.series[ticker],
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: "transparent",
    borderWidth: 1,
    pointRadius: 0,
    tension: 0.1,
  }));

  if (returnsChartInstance) returnsChartInstance.destroy();

  returnsChartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: returnsData.dates, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        title: { display: true, text: "Rendements log journaliers" },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Rendement" } },
      },
    },
  });
}

// --------------------- Matrice de corrélation ---------------------
// Implémentée en table HTML colorée (pas besoin du plugin chartjs-chart-matrix,
// donc aucune dépendance supplémentaire par rapport à votre stack actuelle).
function renderCorrelationMatrix(correlationData) {
  const container = document.getElementById("correlationMatrix");
  container.innerHTML = "";

  const { labels, matrix } = correlationData;

  const table = document.createElement("table");
  table.className = "table table-bordered table-sm text-center correlation-table";

  // En-tête
  const headerRow = document.createElement("tr");
  headerRow.appendChild(document.createElement("th"));
  labels.forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  // Lignes
  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.textContent = labels[i];
    tr.appendChild(rowHeader);

    row.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value.toFixed(2);
      td.style.backgroundColor = correlationColor(value);
      td.style.color = Math.abs(value) > 0.6 ? "#fff" : "#000";
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  container.appendChild(table);
}

// Échelle de couleur divergente : rouge (-1) -> blanc (0) -> bleu (+1)
function correlationColor(value) {
  const v = Math.max(-1, Math.min(1, value));
  if (v >= 0) {
    const intensity = Math.round(255 * (1 - v));
    return `rgb(${intensity}, ${intensity}, 255)`;
  } else {
    const intensity = Math.round(255 * (1 + v));
    return `rgb(255, ${intensity}, ${intensity})`;
  }
}