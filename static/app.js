const btn = document.getElementById("predict-btn");
const gpSelect = document.getElementById("gp-select");
const qualiList = document.getElementById("quali-list");
const raceList = document.getElementById("race-list");
const statusEl = document.getElementById("prediction-status");

function safePct(value) {
  return Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));
}

function predictionRows(entries, probKey) {
  return entries.map(e => {
    const pct = safePct(e[probKey]);
    const lap = e.predicted_final_lap == null
      ? ""
      : `<span class="final-lap">Final lap ${Number(e.predicted_final_lap).toFixed(3)}s</span>`;
    return `
      <li class="rank-row">
        <div class="rank-driver">
          <span class="code">P${e.position} ${e.driver}</span>
          <span class="team">${e.team}</span>
          ${lap}
        </div>
        <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
        <span class="rank-prob">${pct}%</span>
      </li>`;
  }).join("");
}

if (btn) {
  btn.addEventListener("click", async () => {
    const gp = gpSelect.value;
    if (!gp) return;

    btn.disabled = true;
    btn.textContent = "Collecting FastF1 data...";
    statusEl.textContent = "Collecting 2025 history and completed 2026 results. The first run can take several minutes.";
    qualiList.innerHTML = `<li class="placeholder">Building 2026 driver form...</li>`;
    raceList.innerHTML = `<li class="placeholder">Waiting for qualifying prediction...</li>`;

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ year: 2026, gp })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed.");

      qualiList.innerHTML = predictionRows(data.quali, "quali_top10_prob");
      raceList.innerHTML = predictionRows(data.race, "race_top10_prob");

      const lapStatus = data.final_lap?.available
        ? ` Final-lap forecast uses ${data.final_lap.historical_year} ${data.gp} telemetry.`
        : ` ${data.final_lap?.message || "Final-lap forecast unavailable."}`;
      statusEl.textContent = `${data.gp} 2026 — history available before Round ${data.target_round}.${lapStatus}`;
    } catch (err) {
      qualiList.innerHTML = `<li class="placeholder">${err.message}</li>`;
      raceList.innerHTML = `<li class="placeholder">Prediction failed.</li>`;
      statusEl.textContent = "";
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict 2026 Race";
    }
  });
}
