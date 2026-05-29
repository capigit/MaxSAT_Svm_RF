const report = window.LAD_REPORT_DATA;

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
