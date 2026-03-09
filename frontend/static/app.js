let currentJobId = null;
let statusInterval = null;
let currentConfig = null;

// Tab switching
function showTab(tabName, buttonElement) {
  document.querySelectorAll(".tab-content").forEach((tab) => {
    tab.classList.remove("active");
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
  });

  document.getElementById(`${tabName}-tab`).classList.add("active");

  if (buttonElement) {
    buttonElement.classList.add("active");
  } else {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      if (btn.textContent.toLowerCase().includes(tabName)) {
        btn.classList.add("active");
      }
    });
  }

  if (tabName === "results") {
    loadResults();
  }

  // When switching back to Run tab with an active job, refresh status immediately
  // so the progress bar displays correctly (avoids stale display after tab was hidden)
  if (tabName === "run" && currentJobId) {
    fetch(`/api/experiments/status/${currentJobId}`)
      .then((r) => r.json())
      .then((status) => {
        if (!status.error) updateStatusDisplay(status);
      })
      .catch(() => {});
  }
}

// Configuration management
async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    currentConfig = config;
    renderConfig(config);
    showMessage("Configuration loaded", "success");
  } catch (error) {
    showMessage("Error loading configuration: " + error.message, "error");
  }
}

function renderConfig(config) {
  // Global settings
  document.getElementById("max-parallel").value =
    config.global_settings?.max_parallel_experiments || 2;
  document.getElementById("output-dir").value =
    config.global_settings?.output_dir || "./experiments_results";
  document.getElementById("log-level").value =
    config.global_settings?.log_level || "INFO";

  // Organoid experiments
  const organoidContainer = document.getElementById("organoid-experiments");
  organoidContainer.innerHTML = "";
  (config.organoid_experiments || []).forEach((exp, idx) => {
    organoidContainer.appendChild(createOrganoidExperimentForm(exp, idx));
  });

  // TIL experiments
  const tilContainer = document.getElementById("til-experiments");
  tilContainer.innerHTML = "";
  (config.til_experiments || []).forEach((exp, idx) => {
    tilContainer.appendChild(createTILExperimentForm(exp, idx));
  });
}

function createOrganoidExperimentForm(exp, idx) {
  const div = document.createElement("div");
  div.className = "experiment-form";
  div.innerHTML = `
        <div class="experiment-header">
            <h4>Experiment ${idx + 1}: ${exp.name || "Unnamed"}</h4>
            <button class="btn-remove" onclick="removeOrganoidExperiment(${idx})">×</button>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Name:</label>
                <input type="text" class="form-control exp-name" value="${exp.name || ""}" 
                       onchange="updateOrganoidExperiment(${idx}, 'name', this.value)">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Model Type:</label>
                <select class="form-control" onchange="updateOrganoidExperiment(${idx}, 'model_type', this.value)">
                    <option value="fusion" ${exp.model_type === "fusion" || !exp.model_type ? "selected" : ""}>Fusion (LSTM + MLP)</option>
                    <option value="random_forest" ${exp.model_type === "random_forest" ? "selected" : ""}>Random Forest</option>
                </select>
            </div>
            <div class="form-group">
                <label>Dataset:</label>
                <select class="form-control" onchange="updateOrganoidExperiment(${idx}, 'dataset', this.value)">
                    <option value="all" ${exp.dataset === "all" || !exp.dataset ? "selected" : ""}>All Datasets</option>
                    <option value="2ND" ${exp.dataset === "2ND" ? "selected" : ""}>2ND</option>
                    <option value="CAF" ${exp.dataset === "CAF" ? "selected" : ""}>CAF</option>
                    <option value="CART" ${exp.dataset === "CART" ? "selected" : ""}>CART</option>
                    <option value="PDO" ${exp.dataset === "PDO" ? "selected" : ""}>PDO</option>
                </select>
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Hidden Sizes (comma-separated):</label>
                <input type="text" class="form-control" 
                       value="${Array.isArray(exp.hidden_sizes) ? exp.hidden_sizes.join(", ") : exp.hidden_sizes || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'hidden_sizes', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Fusion Sizes (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.fusion_sizes) ? exp.fusion_sizes.join(", ") : exp.fusion_sizes || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'fusion_sizes', parseArray(this.value))">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Dropout (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.dropout) ? exp.dropout.join(", ") : exp.dropout || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'dropout', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Sequence Length (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.seq_len) ? exp.seq_len.join(", ") : exp.seq_len || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'seq_len', parseArray(this.value))">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Batch Size (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.batch_size) ? exp.batch_size.join(", ") : exp.batch_size || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'batch_size', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Epochs (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.epochs) ? exp.epochs.join(", ") : exp.epochs || ""}"
                       onchange="updateOrganoidExperiment(${idx}, 'epochs', parseArray(this.value))">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Features:</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.features) ? exp.features.join(", ") : exp.features || "all"}"
                       onchange="updateOrganoidExperiment(${idx}, 'features', parseArrayOrString(this.value))">
            </div>
            <div class="form-group">
                <label>Track Features:</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.track_features) ? exp.track_features.join(", ") : exp.track_features || "all"}"
                       onchange="updateOrganoidExperiment(${idx}, 'track_features', parseArrayOrString(this.value))">
            </div>
        </div>
    `;
  return div;
}

function createTILExperimentForm(exp, idx) {
  const div = document.createElement("div");
  div.className = "experiment-form";
  div.innerHTML = `
        <div class="experiment-header">
            <h4>Experiment ${idx + 1}: ${exp.name || "Unnamed"}</h4>
            <button class="btn-remove" onclick="removeTILExperiment(${idx})">×</button>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Name:</label>
                <input type="text" class="form-control exp-name" value="${exp.name || ""}"
                       onchange="updateTILExperiment(${idx}, 'name', this.value)">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Model Architecture:</label>
                <select class="form-control" onchange="updateTILExperiment(${idx}, 'model_type', this.value)">
                    <option value="resnet18" ${exp.model_type === "resnet18" || !exp.model_type ? "selected" : ""}>ResNet-18 (Default)</option>
                    <option value="resnet34" ${exp.model_type === "resnet34" ? "selected" : ""}>ResNet-34</option>
                    <option value="resnet50" ${exp.model_type === "resnet50" ? "selected" : ""}>ResNet-50</option>
                </select>
            </div>
            <div class="form-group">
                <label>Dataset:</label>
                <select class="form-control" onchange="updateTILExperiment(${idx}, 'dataset', this.value)">
                    <option value="chip" ${exp.dataset === "chip" || !exp.dataset ? "selected" : ""}>Chip Images (Default)</option>
                    <option value="clinical" ${exp.dataset === "clinical" ? "selected" : ""}>Clinical Images</option>
                </select>
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Intermediate Features (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.intermediate_features) ? exp.intermediate_features.join(", ") : exp.intermediate_features || ""}"
                       onchange="updateTILExperiment(${idx}, 'intermediate_features', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Second Intermediate Features (comma-separated, optional):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.second_intermediate_features) ? exp.second_intermediate_features.join(", ") : exp.second_intermediate_features || ""}"
                       onchange="updateTILExperiment(${idx}, 'second_intermediate_features', parseArray(this.value))">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Dropout Prob (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.dropout_prob) ? exp.dropout_prob.join(", ") : exp.dropout_prob || ""}"
                       onchange="updateTILExperiment(${idx}, 'dropout_prob', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Batch Size (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.batch_size) ? exp.batch_size.join(", ") : exp.batch_size || ""}"
                       onchange="updateTILExperiment(${idx}, 'batch_size', parseArray(this.value))">
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Epochs (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.epochs) ? exp.epochs.join(", ") : exp.epochs || ""}"
                       onchange="updateTILExperiment(${idx}, 'epochs', parseArray(this.value))">
            </div>
            <div class="form-group">
                <label>Learning Rate (comma-separated):</label>
                <input type="text" class="form-control"
                       value="${Array.isArray(exp.learning_rate) ? exp.learning_rate.join(", ") : exp.learning_rate || ""}"
                       onchange="updateTILExperiment(${idx}, 'learning_rate', parseArray(this.value))">
            </div>
        </div>
    `;
  return div;
}

function parseArray(value) {
  if (!value || value.trim() === "") return [];
  return value.split(",").map((v) => {
    const trimmed = v.trim();
    const num = parseFloat(trimmed);
    return isNaN(num) ? trimmed : num;
  });
}

function parseArrayOrString(value) {
  if (!value || value.trim() === "") return "all";
  const trimmed = value.trim();
  if (trimmed === "all") return "all";
  const arr = parseArray(value);
  return arr.length === 1 ? arr[0] : arr;
}

function updateOrganoidExperiment(idx, key, value) {
  if (!currentConfig.organoid_experiments)
    currentConfig.organoid_experiments = [];
  if (!currentConfig.organoid_experiments[idx])
    currentConfig.organoid_experiments[idx] = {};
  currentConfig.organoid_experiments[idx][key] = value;
}

function updateTILExperiment(idx, key, value) {
  if (!currentConfig.til_experiments) currentConfig.til_experiments = [];
  if (!currentConfig.til_experiments[idx])
    currentConfig.til_experiments[idx] = {};
  currentConfig.til_experiments[idx][key] = value;
  if (value === "" || (Array.isArray(value) && value.length === 0)) {
    delete currentConfig.til_experiments[idx][key];
  }
}

function removeOrganoidExperiment(idx) {
  if (confirm("Remove this experiment?")) {
    currentConfig.organoid_experiments.splice(idx, 1);
    renderConfig(currentConfig);
  }
}

function removeTILExperiment(idx) {
  if (confirm("Remove this experiment?")) {
    currentConfig.til_experiments.splice(idx, 1);
    renderConfig(currentConfig);
  }
}

function addOrganoidExperiment() {
  if (!currentConfig.organoid_experiments)
    currentConfig.organoid_experiments = [];
  currentConfig.organoid_experiments.push({
    name: `organoid_experiment_${currentConfig.organoid_experiments.length + 1}`,
    model_type: "fusion",
    dataset: "all",
    hidden_sizes: [32],
    fusion_sizes: [64],
    dropout: [0.3],
    seq_len: [100],
    batch_size: [256],
    epochs: [200],
    features: "all",
    track_features: "all",
  });
  renderConfig(currentConfig);
}

function addTILExperiment() {
  if (!currentConfig.til_experiments) currentConfig.til_experiments = [];
  currentConfig.til_experiments.push({
    name: `til_experiment_${currentConfig.til_experiments.length + 1}`,
    model_type: "resnet18",
    dataset: "chip",
    intermediate_features: [512],
    dropout_prob: [0.5],
    batch_size: [16],
    epochs: [200],
    learning_rate: [0.00025],
  });
  renderConfig(currentConfig);
}

async function saveConfig() {
  try {
    // Update global settings
    currentConfig.global_settings = {
      max_parallel_experiments: parseInt(
        document.getElementById("max-parallel").value,
      ),
      output_dir: document.getElementById("output-dir").value,
      log_level: document.getElementById("log-level").value,
    };

    const response = await fetch("/api/config", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(currentConfig),
    });

    const result = await response.json();

    if (result.success) {
      showMessage("Configuration saved successfully!", "success");
    } else {
      showMessage("Error saving configuration: " + result.error, "error");
    }
  } catch (error) {
    showMessage("Error: " + error.message, "error");
  }
}

function showMessage(message, type) {
  const messageEl = document.getElementById("config-message");
  messageEl.textContent = message;
  messageEl.className = `message ${type}`;

  setTimeout(() => {
    messageEl.className = "message";
  }, 3000);
}

// Experiment running
async function startExperiments() {
  const analyzer = document.getElementById("analyzer-select").value;
  const testMode = document.getElementById("test-mode").checked;

  try {
    const response = await fetch("/api/experiments/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        analyzer: analyzer,
        test: testMode,
      }),
    });

    const result = await response.json();

    if (result.success) {
      currentJobId = result.job_id;
      document.getElementById("experiment-status").style.display = "block";
      startStatusPolling();
      showTab("run", document.querySelectorAll(".tab-btn")[1]);
    } else {
      alert("Error starting experiments: " + result.error);
    }
  } catch (error) {
    alert("Error: " + error.message);
  }
}

function startStatusPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
  }

  statusInterval = setInterval(async () => {
    if (!currentJobId) return;

    try {
      const response = await fetch(`/api/experiments/status/${currentJobId}`);
      const status = await response.json();

      if (status.error) {
        clearInterval(statusInterval);
        return;
      }

      updateStatusDisplay(status);

      if (status.status === "completed" || status.status === "error") {
        clearInterval(statusInterval);
        loadResults();
      }
    } catch (error) {
      console.error("Error polling status:", error);
    }
  }, 1000);
}

function updateStatusDisplay(status) {
  document.getElementById("status-text").textContent = status.status;
  document.getElementById("current-experiment").textContent =
    status.current || "-";
  document.getElementById("progress-text").textContent =
    `${status.completed}/${status.total}`;
  document.getElementById("completed-count").textContent = status.completed;
  document.getElementById("failed-count").textContent = status.failed;

  const progressFill = document.getElementById("progress-fill");
  progressFill.style.width = `${status.progress}%`;
  progressFill.textContent = `${status.progress}%`;

  const logsEl = document.getElementById("logs");
  logsEl.innerHTML = "";
  status.logs.slice(-50).forEach((log) => {
    const logEntry = document.createElement("div");
    logEntry.className = "log-entry";
    if (log.includes("✅")) {
      logEntry.classList.add("success");
    } else if (log.includes("❌")) {
      logEntry.classList.add("error");
    }
    logEntry.textContent = log;
    logsEl.appendChild(logEntry);
  });
  logsEl.scrollTop = logsEl.scrollHeight;
}

// Results loading
async function loadResults() {
  try {
    const response = await fetch("/api/results");
    const results = await response.json();

    displayResults(results);

    // Load summary graph
    const graphImg = document.getElementById("summary-graph");
    const graphContainer = document.getElementById("summary-graph-container");
    graphImg.src = "/api/results/summary?" + new Date().getTime();
    graphImg.onload = () => {
      graphContainer.style.display = "block";
    };
    graphImg.onerror = () => {
      graphContainer.style.display = "none";
    };
  } catch (error) {
    console.error("Error loading results:", error);
  }
}

function displayResults(results) {
  const organoidEl = document.getElementById("organoid-results");
  const tilEl = document.getElementById("til-results");

  if (results.organoid && results.organoid.length > 0) {
    organoidEl.innerHTML = results.organoid
      .map((result) => {
        const date = new Date(result.timestamp * 1000).toLocaleString();
        return `
                <div class="result-item">
                    <h4>${result.name}</h4>
                    <div class="result-value">Test Accuracy: ${result.test_accuracy.toFixed(4)}</div>
                    <div class="result-meta">Last updated: ${date}</div>
                </div>
            `;
      })
      .join("");
  } else {
    organoidEl.innerHTML =
      '<div class="empty-state">No Organoid results yet</div>';
  }

  if (results.til && results.til.length > 0) {
    tilEl.innerHTML = results.til
      .map((result) => {
        const date = new Date(result.timestamp * 1000).toLocaleString();
        const metrics = result.metrics || {};
        return `
                <div class="result-item">
                    <h4>${result.name}</h4>
                    ${metrics.quality_score ? `<div class="result-value">Quality Score: ${metrics.quality_score.toFixed(3)}</div>` : ""}
                    ${metrics.survival_discrimination ? `<div class="result-meta">Discrimination: ${metrics.survival_discrimination.toFixed(3)}</div>` : ""}
                    <div class="result-meta">Last updated: ${date}</div>
                </div>
            `;
      })
      .join("");
  } else {
    tilEl.innerHTML = '<div class="empty-state">No TIL results yet</div>';
  }
}

// ─── GigaTIME Tab ────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  loadResults();

  const tileInput = document.getElementById("he-tile-input");
  if (tileInput) {
    tileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const preview = document.getElementById("he-preview");
        preview.src = e.target.result;
        document.getElementById("he-preview-container").style.display = "block";
      };
      reader.readAsDataURL(file);
    });
  }
});

async function runGigaTIME() {
  const fileInput = document.getElementById("he-tile-input");
  const statusEl = document.getElementById("gigatime-status");
  const runBtn = document.getElementById("gigatime-run-btn");
  const resultsArea = document.getElementById("gigatime-results-area");
  const progressBox = document.getElementById("gigatime-progress");
  const progressFill = document.getElementById("gigatime-progress-fill");
  const stepLabel = document.getElementById("gigatime-step-label");

  if (!fileInput.files || !fileInput.files[0]) {
    statusEl.textContent = "Please select an H&E tile image first.";
    statusEl.className = "gigatime-status error";
    return;
  }

  runBtn.disabled = true;
  statusEl.textContent = "";
  statusEl.className = "gigatime-status";
  resultsArea.style.display = "none";
  progressBox.style.display = "block";
  progressFill.style.width = "0%";
  stepLabel.textContent = "Starting\u2026";

  const formData = new FormData();
  formData.append("tile", fileInput.files[0]);

  try {
    const response = await fetch("/api/gigatime/predict/stream", {
      method: "POST",
      body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const ev = JSON.parse(part.slice(6));
        if (ev.step === 0) {
          progressBox.style.display = "none";
          statusEl.textContent = ev.message;
          statusEl.className = "gigatime-status error";
        } else {
          const pct = Math.round((ev.step / ev.total) * 100);
          progressFill.style.width = pct + "%";
          stepLabel.textContent = `Step ${ev.step}/${ev.total}: ${ev.message}`;
          if (ev.result) {
            progressBox.style.display = "none";
            document.getElementById("gigatime-input-result").src =
              "data:image/png;base64," + ev.result.input;
            const grid = document.getElementById("gigatime-channel-grid");
            grid.innerHTML = ev.result.channels
              .map(
                (ch) => `
              <div class="gigatime-channel-card">
                <img src="data:image/png;base64,${ch.image}" alt="${ch.name}" class="gigatime-channel-img">
                <div class="gigatime-channel-label">${ch.name}</div>
              </div>`,
              )
              .join("");
            resultsArea.style.display = "block";
            statusEl.textContent = "Inference complete \u2014 23 channels generated.";
            statusEl.className = "gigatime-status success";
          }
        }
      }
    }
  } catch (err) {
    progressBox.style.display = "none";
    statusEl.textContent = "Network error: " + err.message;
    statusEl.className = "gigatime-status error";
  } finally {
    runBtn.disabled = false;
  }
}
