const btn = document.getElementById("predict-btn");
const qualiList = document.getElementById("quali-list");
const raceList = document.getElementById("race-list");

function rowsHtml(entries, probKey) {
  return entries.map(e => {
    const pct = Math.round(e[probKey] * 100);
    return `
      <li class="rank-row">
        <div class="rank-driver">
          <span class="code">${e.driver}</span>
          <span class="team">${e.team}</span>
        </div>
        <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
        <span class="rank-prob">${pct}%</span>
      </li>`;
  }).join("");
}

btn.addEventListener("click", async () => {
  btn.disabled = true;
  btn.textContent = "Running...";

  const grid_positions = {};
  document.querySelectorAll(".roster-row").forEach(row => {
    const driver = row.dataset.driver;
    const val = row.querySelector(".grid-pos-input").value;
    grid_positions[driver] = Number(val);
  });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid_positions }),
    });
    const data = await res.json();

    qualiList.innerHTML = rowsHtml(data.quali, "quali_top10_prob");
    raceList.innerHTML = rowsHtml(data.race, "race_top10_prob");
  } catch (err) {
    qualiList.innerHTML = `<li class="placeholder">Prediction failed — check the server console.</li>`;
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run prediction";
  }
});

// --- Final Lap Time panel ---
const lastlapBtn = document.getElementById("lastlap-btn");
const raceSelect = document.getElementById("race-select");
const lastlapTbody = document.getElementById("lastlap-tbody");
const lastlapSummary = document.getElementById("lastlap-summary");

if (lastlapBtn) {
  lastlapBtn.addEventListener("click", async () => {
    const selected = raceSelect.value;
    if (!selected) return;
    const [year, gp, session] = selected.split("|");

    lastlapBtn.disabled = true;
    lastlapBtn.textContent = "Analyzing...";
    lastlapSummary.textContent = "";
    lastlapTbody.innerHTML = `<tr><td colspan="7" class="placeholder">Training model...</td></tr>`;

    try {
      const res = await fetch("/predict_lastlap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ year: Number(year), gp, session }),
      });
      const data = await res.json();

      if (!res.ok) {
        lastlapTbody.innerHTML = `<tr><td colspan="7" class="placeholder">${data.error || "Prediction failed."}</td></tr>`;
        lastlapSummary.textContent = "";
        return;
      }

      lastlapSummary.textContent =
        `Best model: ${data.best_model}  ·  MAE ${data.mae}s  ·  RMSE ${data.rmse}s  ·  R² ${data.r2}`;

      lastlapTbody.innerHTML = data.drivers.map(d => {
        const errClass = d.error >= 0 ? "error-positive" : "error-negative";
        const errSign = d.error >= 0 ? "+" : "";
        return `
          <tr>
            <td>${d.driver}</td>
            <td>${d.team}</td>
            <td>${d.compound}</td>
            <td class="mono">${d.tyre_life}</td>
            <td class="mono">${d.actual.toFixed(3)}s</td>
            <td class="mono">${d.predicted.toFixed(3)}s</td>
            <td class="mono ${errClass}">${errSign}${d.error.toFixed(3)}s</td>
          </tr>`;
      }).join("");
    } catch (err) {
      lastlapTbody.innerHTML = `<tr><td colspan="7" class="placeholder">Prediction failed — check the server console.</td></tr>`;
      console.error(err);
    } finally {
      lastlapBtn.disabled = false;
      lastlapBtn.textContent = "Analyze final laps";
    }
  });
}
