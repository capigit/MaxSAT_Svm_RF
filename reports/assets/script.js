const report = window.LAD_REPORT_DATA;

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
