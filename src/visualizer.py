import json
from pathlib import Path


def generer_rapport_resultats(resultats, dossier_sortie='reports'):
    """Génère un rapport web statique HTML/CSS/JS pour les résultats."""
    output_dir = Path(dossier_sortie)
    assets_dir = output_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / '.nojekyll').write_text('', encoding='utf-8')
    (output_dir / 'index.html').write_text(_html(), encoding='utf-8')
    (assets_dir / 'style.css').write_text(_css(), encoding='utf-8')
    (assets_dir / 'script.js').write_text(_js(), encoding='utf-8')
    (assets_dir / 'data.js').write_text(_data_js(resultats), encoding='utf-8')

    chemin_rapport = output_dir / 'index.html'
    print(f"[Succès] Le rapport web a été sauvegardé sous le nom : '{chemin_rapport}'")


def _data_js(resultats):
    contenu_json = json.dumps(resultats, ensure_ascii=False, indent=2)
    return f"window.LAD_REPORT_DATA = {contenu_json};\n"


def _html():
    return """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Rapport LAD-ML : comparaison SVM, LAD-SVM, forêt aléatoire et RF-LAD avec sélection de variables par MaxSAT.">
  <meta name="theme-color" content="#315f9f">
  <title>Rapport LAD-ML</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <main class="page-shell">
    <header class="report-header">
      <div>
        <p class="eyebrow">Projet LAD-ML</p>
        <h1>Sélection de variables par MaxSAT</h1>
      </div>
      <div class="header-meta">
        <span id="dataset-name">Dataset</span>
        <strong id="generated-at">Rapport</strong>
      </div>
    </header>

    <section class="summary-grid" aria-label="Synthèse">
      <article class="metric-card">
        <span>Exemples</span>
        <strong id="sample-count">0</strong>
      </article>
      <article class="metric-card">
        <span>Variables initiales</span>
        <strong id="feature-count">0</strong>
      </article>
      <article class="metric-card">
        <span>Variables LAD-SVM</span>
        <strong id="lad-feature-count">0</strong>
      </article>
      <article class="metric-card">
        <span>Réduction LAD</span>
        <strong id="reduction-rate">0%</strong>
      </article>
    </section>

    <section class="content-grid">
      <section class="panel chart-panel">
        <div class="panel-heading">
          <h2>Accuracy par modèle</h2>
          <span>Comparaison classique vs LAD</span>
        </div>
        <div class="chart" id="accuracy-chart" aria-label="Graphique des accuracies"></div>
      </section>

      <section class="panel chart-panel">
        <div class="panel-heading">
          <h2>Variables utilisées</h2>
          <span>Impact de la sélection MaxSAT</span>
        </div>
        <div class="chart" id="feature-chart" aria-label="Graphique des variables"></div>
      </section>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <h2>Tableau des résultats</h2>
        <span>Mesures obtenues sur le jeu de test</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Modèle</th>
              <th>Famille</th>
              <th>Accuracy</th>
              <th>Variables</th>
            </tr>
          </thead>
          <tbody id="results-table"></tbody>
        </table>
      </div>
    </section>

    <section class="content-grid">
      <section class="panel">
        <div class="panel-heading">
          <h2>MSS LAD-SVM</h2>
          <span>Indices des variables conservées</span>
        </div>
        <div class="chips" id="svm-mss"></div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <h2>MSS RF-LAD</h2>
          <span>Ensembles énumérés par clauses bloquantes</span>
        </div>
        <div class="mss-list" id="rf-mss-list"></div>
      </section>
    </section>
  </main>

  <script src="assets/data.js"></script>
  <script src="assets/script.js"></script>
</body>
</html>
"""


def _css():
    return """:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --surface: #ffffff;
  --text: #19202a;
  --muted: #697386;
  --line: #d9dee7;
  --blue: #315f9f;
  --green: #3f8f63;
  --amber: #b7791f;
  --ink: #243447;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
}

.page-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 40px;
}

.report-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding: 18px 0 24px;
  border-bottom: 1px solid var(--line);
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--blue);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 34px;
  line-height: 1.12;
}

h2 {
  font-size: 18px;
}

.header-meta {
  display: grid;
  gap: 5px;
  color: var(--muted);
  text-align: right;
}

.header-meta strong {
  color: var(--text);
  font-size: 14px;
}

.summary-grid,
.content-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.content-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.metric-card,
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.metric-card {
  display: grid;
  gap: 8px;
  min-height: 104px;
  padding: 18px;
}

.metric-card span,
.panel-heading span {
  color: var(--muted);
  font-size: 13px;
}

.metric-card strong {
  font-size: 28px;
  line-height: 1;
}

.panel {
  margin-top: 18px;
  padding: 18px;
}

.content-grid .panel {
  margin-top: 0;
}

.panel-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}

.chart {
  display: grid;
  gap: 12px;
}

.bar-row {
  display: grid;
  grid-template-columns: 132px 1fr 72px;
  align-items: center;
  gap: 12px;
  min-height: 34px;
}

.bar-label {
  color: var(--ink);
  font-size: 14px;
  font-weight: 700;
}

.bar-track {
  height: 16px;
  overflow: hidden;
  background: #e8edf3;
  border-radius: 999px;
}

.bar-fill {
  width: var(--value);
  height: 100%;
  background: var(--blue);
  border-radius: inherit;
}

.bar-fill.lad {
  background: var(--green);
}

.bar-value {
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
  text-align: right;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 680px;
}

th,
td {
  padding: 12px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}

td {
  font-size: 14px;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  min-height: 30px;
  padding: 6px 10px;
  color: var(--ink);
  background: #edf4ef;
  border: 1px solid #c9dfd0;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
}

.mss-list {
  display: grid;
  gap: 10px;
  max-height: 320px;
  overflow: auto;
  padding-right: 4px;
}

.mss-item {
  display: grid;
  gap: 6px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
}

.mss-item strong {
  font-size: 13px;
}

.mss-values {
  color: var(--muted);
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
  line-height: 1.5;
}

@media (max-width: 880px) {
  .report-header,
  .panel-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .header-meta {
    text-align: left;
  }

  .summary-grid,
  .content-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 600px) {
  .page-shell {
    width: min(100% - 20px, 1180px);
    padding-top: 16px;
  }

  h1 {
    font-size: 28px;
  }

  .summary-grid,
  .content-grid {
    grid-template-columns: 1fr;
  }

  .bar-row {
    grid-template-columns: 1fr;
    gap: 6px;
  }

  .bar-value {
    text-align: left;
  }
}
"""


def _js():
    return """const report = window.LAD_REPORT_DATA;

const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
const formatNumber = (value) => Number.isInteger(value) ? `${value}` : `${value}`;

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function renderSummary() {
  const dataset = report.dataset;
  setText('dataset-name', dataset.nom);
  setText('generated-at', `Généré le ${report.date_generation}`);
  setText('sample-count', dataset.nb_exemples);
  setText('feature-count', dataset.nb_variables);
  setText('lad-feature-count', report.mss_svm.length);

  const reduction = 1 - report.mss_svm.length / dataset.nb_variables;
  setText('reduction-rate', `${(reduction * 100).toFixed(1)}%`);
}

function renderChart(containerId, items, unit, maxValue) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = item.nom;

    const track = document.createElement('div');
    track.className = 'bar-track';

    const fill = document.createElement('div');
    fill.className = `bar-fill ${item.lad ? 'lad' : ''}`;
    fill.style.setProperty('--value', `${Math.max(0, item.valeur / maxValue * 100)}%`);

    const value = document.createElement('div');
    value.className = 'bar-value';
    value.textContent = unit === '%' ? `${item.valeur.toFixed(2)}%` : `${formatNumber(item.valeur)} var.`;

    track.appendChild(fill);
    row.append(label, track, value);
    container.appendChild(row);
  });
}

function renderResultsTable() {
  const tbody = document.getElementById('results-table');
  tbody.innerHTML = '';

  report.modeles.forEach((modele) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${modele.nom}</strong></td>
      <td>${modele.famille}</td>
      <td>${formatPercent(modele.accuracy)}</td>
      <td>${modele.variables}</td>
    `;
    tbody.appendChild(row);
  });
}

function renderMss() {
  const svmMss = document.getElementById('svm-mss');
  svmMss.innerHTML = '';
  report.mss_svm.forEach((indice) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = indice;
    svmMss.appendChild(chip);
  });

  const rfList = document.getElementById('rf-mss-list');
  rfList.innerHTML = '';
  report.mss_rf.forEach((mss, index) => {
    const item = document.createElement('article');
    item.className = 'mss-item';
    item.innerHTML = `
      <strong>MSS ${index + 1} - ${mss.length} variables</strong>
      <span class="mss-values">[${mss.join(', ')}]</span>
    `;
    rfList.appendChild(item);
  });
}

function render() {
  renderSummary();

  renderChart(
    'accuracy-chart',
    report.modeles.map((modele) => ({
      nom: modele.nom,
      valeur: modele.accuracy * 100,
      lad: modele.lad
    })),
    '%',
    100
  );

  renderChart(
    'feature-chart',
    report.modeles.map((modele) => ({
      nom: modele.nom,
      valeur: Number.parseFloat(`${modele.variables}`.replace('~', '')),
      lad: modele.lad
    })),
    'var',
    report.dataset.nb_variables
  );

  renderResultsTable();
  renderMss();
}

render();
"""
