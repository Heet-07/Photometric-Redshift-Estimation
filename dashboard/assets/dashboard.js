const get = (id) => document.getElementById(id);
const formatCount = new Intl.NumberFormat("en-US");
const estimateForm = get("estimate-form");
const bandInputs = [...estimateForm.querySelectorAll("input")];
const predictionValue = get("prediction-value");
const predictionNote = get("prediction-note");
const sample = [21.32676, 19.57066, 18.02653, 17.46703, 17.04413];

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({}));
    throw new Error(typeof detail === "string" ? detail : "The server could not complete the request.");
  }
  return response;
}

function renderDistribution(bins) {
  const maximum = Math.max(...bins.map(({ count }) => count));
  const scale = Math.max(4, Math.ceil(maximum / 2000) * 2000);
  for (const tick of [scale, scale * .75, scale * .5, scale * .25, 0]) {
    const label = document.createElement("span");
    label.textContent = formatCount.format(tick);
    get("chart-scale").append(label);
  }
  for (const { lower, upper, count } of bins) {
    const range = lower.toFixed(3) + "–" + upper.toFixed(3);
    const bar = document.createElement("span");
    bar.style.height = (count / scale * 100) + "%";
    bar.title = range + " · " + formatCount.format(count) + " galaxies";
    get("bars").append(bar);

    const row = get("bin-table").insertRow();
    row.insertCell().textContent = range;
    row.insertCell().textContent = formatCount.format(count);
  }
  get("chart-message").hidden = true;
  get("distribution").hidden = false;
}

function renderCharts(container, charts) {
  for (const { name, title, description } of charts) {
    const figure = document.createElement("figure");
    figure.className = "chart chart-card";
    const caption = document.createElement("figcaption");
    const heading = document.createElement("h3");
    heading.className = "chart-title";
    heading.textContent = title;
    const note = document.createElement("p");
    note.textContent = description;
    caption.append(heading, note);

    const image = document.createElement("img");
    image.alt = title;
    image.loading = "lazy";
    image.decoding = "async";
    const url = "/dashboard-charts/" + encodeURIComponent(name);
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    link.setAttribute("aria-label", "Open full-size figure: " + title);
    link.append(image);
    image.addEventListener("error", () => {
      link.hidden = true;
      note.textContent = "This figure is unavailable.";
    });
    image.src = url;
    figure.append(caption, link);
    container.append(figure);
  }
}

async function loadOverview() {
  const evaluationNote = get("evaluation-note");
  const reportsNote = get("reports-note");
  try {
    const response = await request("/dashboard-data");
    const data = await response.json();
    if (Number.isInteger(data.test_rows)) {
      get("test-rows").textContent = formatCount.format(data.test_rows);
    }
    for (const name of ["RMSE", "MAE", "R2"]) {
      const value = data.test_metrics[name];
      if (Number.isFinite(value)) get(name.toLowerCase()).textContent = value.toFixed(5);
    }
    evaluationNote.textContent = Number.isInteger(data.training_rows)
      ? "Training sample: " + formatCount.format(data.training_rows) + " galaxies."
      : "Training sample size unavailable.";
    renderCharts(get("diagnostic-charts"), data.diagnostic_charts ?? []);
    const reports = data.report_figures ?? [];
    renderCharts(get("report-figures"), reports);
    reportsNote.hidden = reports.length > 0;
    reportsNote.textContent = "No saved project figures were found.";
    if (data.redshift_distribution.length) {
      renderDistribution(data.redshift_distribution);
    } else {
      get("chart-message").textContent = "Run the project pipeline to generate the held-out results.";
    }
  } catch {
    evaluationNote.textContent = "Evaluation metrics could not be loaded.";
    get("chart-message").textContent = "Evaluation results could not be loaded.";
    reportsNote.textContent = "Saved project figures could not be loaded.";
  }
}

function clearPrediction() {
  predictionValue.textContent = "—";
  predictionNote.textContent = "The evaluation sample covers z = 0.01–0.50.";
  delete predictionNote.dataset.state;
}

get("load-example").addEventListener("click", () => {
  bandInputs.forEach((input, index) => { input.value = sample[index]; });
  clearPrediction();
});
estimateForm.addEventListener("input", clearPrediction);

estimateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const row = Object.fromEntries(bandInputs.map((input) => [input.name, input.valueAsNumber]));
  clearPrediction();
  predictionNote.textContent = "Running model…";

  const controls = [...estimateForm.querySelectorAll("button, input")];
  controls.forEach((control) => { control.disabled = true; });
  try {
    const response = await request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: [row] }),
    });
    const { predictions: [prediction] } = await response.json();
    if (prediction.prediction_status !== "predicted") {
      throw new Error("Check the magnitudes. " + prediction.invalid_reason + ".");
    }
    predictionValue.textContent = prediction.predicted_specz.toFixed(5);
    predictionNote.textContent = "XGBoost estimate from five SDSS magnitudes.";
  } catch (error) {
    predictionNote.textContent = error.message;
    predictionNote.dataset.state = "error";
  } finally {
    controls.forEach((control) => { control.disabled = false; });
  }
});

const batchForm = get("batch-form");
const batchFile = get("batch-file");
const batchFeedback = get("batch-feedback");

batchFile.addEventListener("change", () => {
  batchFeedback.textContent = "";
  delete batchFeedback.dataset.state;
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = batchFile.files[0];
  if (!file) return;

  if (file.size > 2 * 1024 * 1024) {
    batchFeedback.textContent = "The CSV must be 2 MiB or smaller.";
    batchFeedback.dataset.state = "error";
    return;
  }

  const button = batchForm.querySelector("button");
  button.disabled = true;
  batchFile.disabled = true;
  delete batchFeedback.dataset.state;
  batchFeedback.textContent = "Processing CSV…";
  try {
    const response = await request("/predict-csv", {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: file,
    });
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "redshift-predictions.csv";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    const predicted = Number(response.headers.get("X-Predicted-Rows"));
    const total = Number(response.headers.get("X-Input-Rows"));
    batchFeedback.textContent = formatCount.format(predicted) + " predicted · " +
      formatCount.format(total - predicted) + " invalid. CSV downloaded.";
  } catch (error) {
    batchFeedback.textContent = error.message;
    batchFeedback.dataset.state = "error";
  } finally {
    button.disabled = false;
    batchFile.disabled = false;
  }
});

loadOverview();
