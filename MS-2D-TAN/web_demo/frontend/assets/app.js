const apiBase = '/api';

const elements = {
  datasetSelect: document.getElementById('dataset-select'),
  datasetSummary: document.getElementById('dataset-summary'),
  videoSelect: document.getElementById('video-select'),
  refreshVideos: document.getElementById('refresh-videos'),
  videoDuration: document.getElementById('video-duration'),
  examples: document.getElementById('examples-container'),
  description: document.getElementById('description-input'),
  topk: document.getElementById('topk-input'),
  nms: document.getElementById('nms-input'),
  submit: document.getElementById('submit-button'),
  status: document.getElementById('status-message'),
  resultsMeta: document.getElementById('results-meta'),
  resultsBody: document.querySelector('#results-table tbody'),
};

const state = {
  datasets: [],
  selectedDataset: null,
  videoOffset: 0,
  videoLimit: 25,
  totalVideos: 0,
};

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const message = error.detail || error.message || response.statusText;
    throw new Error(message);
  }
  return response.json();
}

function renderDatasetSummary(dataset) {
  if (!dataset) {
    elements.datasetSummary.textContent = '';
    return;
  }
  const metrics = Object.entries(dataset.best_metrics)
    .map(([k, v]) => `${k}: ${(v * 100).toFixed(2)}%`)
    .join(' · ');
  const notes = dataset.notes && dataset.notes.length ? ` Notes: ${dataset.notes.join(' ')}` : '';
  elements.datasetSummary.textContent = `${dataset.display_name} — ${dataset.description} (${metrics}).${notes}`;
}

async function loadDatasets() {
  try {
    const data = await fetchJSON(`${apiBase}/datasets`);
    state.datasets = data.datasets;
    elements.datasetSelect.innerHTML = '';
    data.datasets.forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.key;
      option.textContent = dataset.display_name;
      elements.datasetSelect.appendChild(option);
    });
    if (data.datasets.length) {
      state.selectedDataset = data.datasets[0];
      elements.datasetSelect.value = state.selectedDataset.key;
      renderDatasetSummary(state.selectedDataset);
      await loadVideos(true);
    }
  } catch (error) {
    elements.status.textContent = `Failed to load datasets: ${error.message}`;
  }
}

function getSelectedDataset() {
  const key = elements.datasetSelect.value;
  return state.datasets.find((item) => item.key === key) || null;
}

async function loadVideos(reset = false) {
  const dataset = getSelectedDataset();
  if (!dataset) return;
  if (reset) {
    state.videoOffset = 0;
  }
  try {
    const data = await fetchJSON(`${apiBase}/datasets/${dataset.key}/videos?limit=${state.videoLimit}&offset=${state.videoOffset}`);
    elements.videoSelect.innerHTML = '';
    data.videos.forEach((vid) => {
      const option = document.createElement('option');
      option.value = vid;
      option.textContent = vid;
      elements.videoSelect.appendChild(option);
    });
    state.totalVideos = data.total_videos;
    if (data.videos.length) {
      elements.videoSelect.value = data.videos[0];
      await loadExamples();
    } else {
      elements.examples.textContent = 'No videos indexed for this dataset.';
    }
  } catch (error) {
    elements.status.textContent = `Failed to load videos: ${error.message}`;
  }
}

async function loadExamples() {
  const dataset = getSelectedDataset();
  const videoId = elements.videoSelect.value;
  if (!dataset || !videoId) return;
  try {
    const data = await fetchJSON(`${apiBase}/datasets/${dataset.key}/videos/${videoId}/examples`);
    elements.videoDuration.textContent = `Duration: ${data.duration.toFixed(2)}s`;
    if (data.examples.length) {
      const examples = data.examples
        .map((example, idx) => {
          const encoded = encodeURIComponent(example.description);
          return `<button class="example" data-text="${encoded}">Example ${idx + 1}</button>`;
        })
        .join(' ');
      elements.examples.innerHTML = `<span>Sample queries:</span> ${examples}`;
    } else {
      elements.examples.textContent = 'No example descriptions available for this video. Try your own query!';
    }
  } catch (error) {
    elements.videoDuration.textContent = '';
    elements.examples.textContent = `Unable to fetch examples: ${error.message}`;
  }
}

function attachExampleHandlers() {
  elements.examples.addEventListener('click', (event) => {
    const target = event.target;
    if (target.matches('button.example')) {
      elements.description.value = decodeURIComponent(target.dataset.text || '');
    }
  });
}

function nextVideoPage() {
  const dataset = getSelectedDataset();
  if (!dataset || !state.totalVideos) return;
  state.videoOffset += state.videoLimit;
  if (state.videoOffset >= state.totalVideos) {
    state.videoOffset = 0;
  }
  loadVideos();
}

async function submitLocalization() {
  const dataset = getSelectedDataset();
  const videoId = elements.videoSelect.value;
  const description = elements.description.value.trim();
  const topK = Number(elements.topk.value) || 5;
  const nmsValue = elements.nms.value.trim();
  const nms = nmsValue ? Number(nmsValue) : null;

  if (!dataset || !videoId || !description) {
    elements.status.textContent = 'Please choose a dataset, select a video, and provide a query sentence.';
    return;
  }

  elements.status.textContent = 'Running inference…';
  elements.submit.disabled = true;

  try {
    const payload = {
      dataset: dataset.key,
      video_id: videoId,
      description,
      top_k: topK,
    };
    if (nms !== null && !Number.isNaN(nms)) {
      payload.nms_threshold = nms;
    }

    const data = await fetchJSON(`${apiBase}/localize`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    renderResults(data);
    elements.status.textContent = 'Success!';
  } catch (error) {
    elements.status.textContent = `Request failed: ${error.message}`;
    elements.resultsBody.innerHTML = '';
    elements.resultsMeta.textContent = '';
  } finally {
    elements.submit.disabled = false;
  }
}

function renderResults(result) {
  if (!result || !result.segments.length) {
    elements.resultsBody.innerHTML = '<tr><td colspan="4">No proposals returned.</td></tr>';
    elements.resultsMeta.textContent = '';
    return;
  }
  elements.resultsMeta.textContent = `Feature clips: ${result.feature_length}, stride ≈ ${result.feature_stride.toFixed(2)}s, video duration ${result.duration.toFixed(2)}s.`;
  const rows = result.segments.map((segment, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td>${segment.start.toFixed(2)}</td>
      <td>${segment.end.toFixed(2)}</td>
      <td>${segment.score.toFixed(4)}</td>
    </tr>
  `);
  elements.resultsBody.innerHTML = rows.join('');
}

function registerEvents() {
  elements.datasetSelect.addEventListener('change', async () => {
    const dataset = getSelectedDataset();
    state.selectedDataset = dataset;
    renderDatasetSummary(dataset);
    await loadVideos(true);
  });

  elements.refreshVideos.addEventListener('click', (event) => {
    event.preventDefault();
    nextVideoPage();
  });

  elements.videoSelect.addEventListener('change', loadExamples);
  elements.submit.addEventListener('click', submitLocalization);
  attachExampleHandlers();
}

registerEvents();
loadDatasets();
