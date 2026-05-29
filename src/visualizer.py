import json
from pathlib import Path


def generer_rapport_resultats(resultats, dossier_sortie='reports'):
    """Génère un rapport web statique HTML/CSS/JS pour les résultats."""
    output_dir = Path(dossier_sortie)
    assets_dir = output_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / '.nojekyll').write_text('', encoding='utf-8')
    (output_dir / 'index.html').write_text(_html(), encoding='utf-8')
    (output_dir / 'favicon.svg').write_text(_favicon(), encoding='utf-8')
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
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <main class="page-shell">
    <header class="report-header">
      <div class="header-copy">
        <p class="eyebrow">Projet LAD-ML</p>
        <h1>Sélection de variables par MaxSAT</h1>
        <p class="lead">Analyse expérimentale de la réduction de variables par logique MaxSAT, appliquée à des classificateurs SVM et forêts aléatoires.</p>
      </div>
      <div class="header-meta">
        <span id="dataset-name">Dataset</span>
        <strong id="generated-at">Rapport</strong>
      </div>
    </header>

    <section class="method-strip" aria-label="Méthode">
      <article>
        <span>1</span>
        <strong>Binarisation</strong>
        <p>Le jeu Breast Cancer est transformé en variables binaires par seuillage.</p>
      </article>
      <article>
        <span>2</span>
        <strong>MaxSAT</strong>
        <p>Les MSS minimisent les variables tout en séparant les classes.</p>
      </article>
      <article>
        <span>3</span>
        <strong>Évaluation</strong>
        <p>Les modèles classiques sont comparés aux variantes LAD.</p>
      </article>
    </section>

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

    <section class="insight-grid" aria-label="Conclusions principales">
      <article class="insight-card primary">
        <span>Observation principale</span>
        <strong id="main-insight">Analyse en cours</strong>
        <p id="main-insight-detail"></p>
      </article>
      <article class="insight-card">
        <span>Gain de compacité</span>
        <strong id="compactness-gain">0 variable</strong>
        <p>Variables retirées par le MSS LAD-SVM par rapport au modèle complet.</p>
      </article>
      <article class="insight-card">
        <span>Écart RF-LAD</span>
        <strong id="rf-gap">0 point</strong>
        <p>Différence d'accuracy entre forêt aléatoire classique et RF-LAD.</p>
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

    <section class="panel results-panel">
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

    <section class="panel conclusion-panel">
      <div class="panel-heading">
        <h2>Conclusion expérimentale</h2>
        <span>Synthèse automatique</span>
      </div>
      <p id="conclusion-text"></p>
    </section>
  </main>

  <script src="assets/data.js"></script>
  <script src="assets/script.js"></script>
</body>
</html>
"""


def _favicon():
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#315f9f"/>
  <path d="M18 41h28" stroke="#ffffff" stroke-width="5" stroke-linecap="round"/>
  <path d="M20 35l8-14 8 14 8-20" fill="none" stroke="#8ee0a3" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="20" cy="35" r="4" fill="#ffffff"/>
  <circle cx="28" cy="21" r="4" fill="#ffffff"/>
  <circle cx="36" cy="35" r="4" fill="#ffffff"/>
  <circle cx="44" cy="15" r="4" fill="#ffffff"/>
</svg>
"""


def _css():
    return """:root {
  color-scheme: light;
  --bg: #f4f6f8;
  --surface: #ffffff;
  --text: #19202a;
  --muted: #697386;
  --line: #d9dee7;
  --blue: #275a92;
  --blue-soft: #e8f0f8;
  --green: #2f855a;
  --green-soft: #e8f5ed;
  --amber: #b7791f;
  --ink: #243447;
  --shadow: 0 16px 42px rgba(31, 43, 58, 0.08);
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
  width: min(1200px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 44px;
}

.report-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding: 24px 0 26px;
  border-bottom: 1px solid var(--line);
}

.header-copy {
  max-width: 760px;
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
  font-size: 38px;
  line-height: 1.12;
}

h2 {
  font-size: 18px;
}

.lead {
  max-width: 700px;
  margin-top: 12px;
  color: var(--muted);
  font-size: 16px;
  line-height: 1.55;
}

.header-meta {
  display: grid;
  gap: 5px;
  min-width: 220px;
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
  text-align: right;
  box-shadow: var(--shadow);
}

.header-meta strong {
  color: var(--text);
  font-size: 14px;
}

.method-strip,
.summary-grid,
.insight-grid,
.content-grid {
  display: grid;
  gap: 14px;
  margin-top: 18px;
}

.method-strip {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.summary-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.insight-grid {
  grid-template-columns: 1.5fr 1fr 1fr;
}

.content-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.method-strip article,
.insight-card,
.metric-card,
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.method-strip article {
  display: grid;
  grid-template-columns: 38px 1fr;
  gap: 5px 12px;
  align-items: center;
  min-height: 106px;
  padding: 16px;
}

.method-strip span {
  grid-row: span 2;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  color: var(--blue);
  background: var(--blue-soft);
  border-radius: 999px;
  font-weight: 700;
}

.method-strip strong {
  color: var(--ink);
  font-size: 14px;
}

.method-strip p {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.metric-card {
  display: grid;
  gap: 8px;
  min-height: 104px;
  padding: 18px;
}

.metric-card span,
.insight-card span,
.panel-heading span {
  color: var(--muted);
  font-size: 13px;
}

.metric-card strong {
  font-size: 28px;
  line-height: 1;
}

.insight-card {
  display: grid;
  align-content: start;
  gap: 9px;
  min-height: 140px;
  padding: 18px;
}

.insight-card.primary {
  background: #f9fbfd;
  border-color: #cfdbea;
}

.insight-card strong {
  color: var(--ink);
  font-size: 21px;
  line-height: 1.25;
}

.insight-card p,
.conclusion-panel p {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.65;
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
  gap: 14px;
}

.bar-row {
  display: grid;
  grid-template-columns: 134px 1fr 78px;
  align-items: center;
  gap: 12px;
  min-height: 38px;
}

.bar-label {
  color: var(--ink);
  font-size: 14px;
  font-weight: 700;
}

.bar-track {
  height: 18px;
  overflow: hidden;
  background: #e8edf3;
  border-radius: 999px;
}

.bar-fill {
  width: var(--value);
  height: 100%;
  background: var(--blue);
  border-radius: inherit;
  transition: width 240ms ease;
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
  border: 1px solid var(--line);
  border-radius: 8px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 680px;
}

th,
td {
  padding: 13px 14px;
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

tbody tr:last-child td {
  border-bottom: 0;
}

tbody tr:nth-child(even) {
  background: #fbfcfd;
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
  background: var(--green-soft);
  border: 1px solid #c9dfd0;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
}

.mss-list {
  display: grid;
  gap: 10px;
  max-height: 360px;
  overflow: auto;
  padding-right: 4px;
}

.mss-item {
  display: grid;
  gap: 6px;
  padding: 11px;
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

  .method-strip,
  .summary-grid,
  .insight-grid,
  .content-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .insight-card.primary {
    grid-column: 1 / -1;
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

  .method-strip,
  .summary-grid,
  .insight-grid,
  .content-grid {
    grid-template-columns: 1fr;
  }

  .method-strip article {
    min-height: auto;
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
const formatPointGap = (value) => `${Math.abs(value).toFixed(2)} pt`;

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function byName(name) {
  return report.modeles.find((modele) => modele.nom === name);
}

function numericVariables(modele) {
  return Number.parseFloat(`${modele.variables}`.replace('~', ''));
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

function renderInsights() {
  const dataset = report.dataset;
  const svm = byName('SVM Classique');
  const ladSvm = byName('LAD-SVM');
  const rf = byName('RF Classique');
  const rfLad = byName('RF-LAD');

  const reductionCount = dataset.nb_variables - numericVariables(ladSvm);
  const reductionRate = reductionCount / dataset.nb_variables * 100;
  const svmGap = (ladSvm.accuracy - svm.accuracy) * 100;
  const rfGap = (rfLad.accuracy - rf.accuracy) * 100;

  const mainLabel = svmGap >= 0
    ? 'LAD-SVM améliore légèrement le SVM classique'
    : 'LAD-SVM reste proche du SVM classique';

  const mainDetail = svmGap >= 0
    ? `Avec ${numericVariables(ladSvm)} variables au lieu de ${dataset.nb_variables}, LAD-SVM gagne ${formatPointGap(svmGap)} d'accuracy sur cette exécution.`
    : `Avec ${numericVariables(ladSvm)} variables au lieu de ${dataset.nb_variables}, LAD-SVM perd seulement ${formatPointGap(svmGap)} d'accuracy sur cette exécution.`;

  setText('main-insight', mainLabel);
  setText('main-insight-detail', mainDetail);
  setText('compactness-gain', `${reductionCount} variables`);
  setText('rf-gap', rfGap >= 0 ? `+${rfGap.toFixed(2)} pt` : `-${Math.abs(rfGap).toFixed(2)} pt`);

  const rfSentence = rfGap >= 0
    ? `RF-LAD obtient une accuracy supérieure de ${formatPointGap(rfGap)} à la forêt classique.`
    : `RF-LAD perd ${formatPointGap(rfGap)} par rapport à la forêt classique, avec des MSS d'environ ${numericVariables(rfLad)} variables.`;

  setText(
    'conclusion-text',
    `La sélection MaxSAT réduit l'espace de ${dataset.nb_variables} à ${numericVariables(ladSvm)} variables pour LAD-SVM, soit une réduction de ${reductionRate.toFixed(1)}%. ${mainDetail} ${rfSentence} Le rapport confirme donc l'intérêt du LAD pour produire des modèles plus compacts tout en conservant des performances proches des approches classiques.`
  );
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
  renderInsights();

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
