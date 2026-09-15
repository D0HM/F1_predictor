const btn = document.getElementById("predict-btn");
const gpSelect = document.getElementById("gp-select");
const qualiList = document.getElementById("quali-list");
const raceList = document.getElementById("race-list");
const statusEl = document.getElementById("prediction-status");

function predictionRows(entries, probKey, showLap = false) {
  return entries.map(e => {
    const pct = Math.round(e[probKey] * 100);
    const lap = showLap
      ? (e.final_lap_time == null ? "Final lap: N/A" : `Final lap: ${e.final_lap_time.toFixed(3)}s`)
      : "";
    return `
      <li class="rank-row">
        <div class="rank-driver">
          <span class="code">P${e.position} ${e.driver}</span>
          <span class="team">${e.team}${showLap ? ` · ${lap}` : ""}</span>
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
    btn.textContent = "Updating data...";
    statusEl.textContent = "FastF1 is updating historical and completed 2026 results. The first run can take a while.";
    qualiList.innerHTML = `<li class="placeholder">Building pre-race form...</li>`;
    raceList.innerHTML = `<li class="placeholder">Waiting for predicted qualifying grid...</li>`;

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ year: 2026, gp }),
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Prediction failed.");

      qualiList.innerHTML = predictionRows(data.quali, "quali_top10_prob");
      raceList.innerHTML = predictionRows(data.race, "race_top10_prob", true);
      statusEl.textContent = `${data.gp} 2026 · form calculated automatically from results before Round ${data.target_round} · ${data.drivers_used} current drivers.`;
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
