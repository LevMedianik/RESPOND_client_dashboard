const API_BASE = "";
let chart;

// Универсальный fetch
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Ошибка загрузки ${url}: ${res.status}`);
  return res.json();
}

// === KPI ===
async function loadMetrics() {
  try {
    const metrics = await fetchJSON(`${API_BASE}/metrics?n=12`);
    const last = metrics.data[metrics.data.length - 1];

    document.getElementById("leadsValue").textContent = last.leads;
    document.getElementById("cplValue").textContent = last.cpl.toFixed(2);
    document.getElementById("roiValue").textContent = last.roi.toFixed(3);

    return metrics;
  } catch (err) {
    console.error("Ошибка загрузки KPI:", err);
  }
}

// === График факт + прогноз ===
async function loadForecast(metrics) {
  try {
    const forecast = await fetchJSON(`${API_BASE}/forecast`);

    const monthsHist = metrics.data.map(d => d.month);
    const leadsHist = metrics.data.map(d => d.leads);

    const monthsForecast = forecast.forecast_monthly.map(d => d.month);
    const leadsForecast = forecast.forecast_monthly.map(d => d.leads_forecast);

    const ctx = document.getElementById("leadsChart").getContext("2d");

    if (!chart) {
      chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: [...monthsHist, ...monthsForecast], // весь период до 2025-12
          datasets: [
            {
              label: "Фактический Leads",
              data: leadsHist,
              borderColor: "blue",
              fill: false
            },
            {
              label: "Прогноз Leads",
              // ВАЖНО: соединяем последнюю фактическую точку + все прогнозные
              data: [...Array(monthsHist.length - 1).fill(null), leadsHist[leadsHist.length - 1], ...leadsForecast],
              borderColor: "red",
              borderDash: [5, 5],
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              grace: "5%"
            }
          }
        }
      });
    } else {
      chart.data.labels = [...monthsHist, ...monthsForecast];
      chart.data.datasets[0].data = leadsHist;
      chart.data.datasets[1].data = [...Array(monthsHist.length - 1).fill(null), leadsHist[leadsHist.length - 1], ...leadsForecast];
      chart.update();
    }
  } catch (err) {
    console.error("Ошибка загрузки прогноза:", err);
  }
}


// === Аномалии ===
async function loadAnomalies() {
  try {
    const data = await fetchJSON(`${API_BASE}/anomalies?metric=cpl&k=2.5`);
    const tbody = document.getElementById("anomaliesTable");
    const notes = document.getElementById("anomaliesNotes");

    tbody.innerHTML = "";

    if (data.anomalies.length === 0) {
      tbody.innerHTML = `<tr><td colspan="3" class="text-center text-green-600">Аномалий не найдено ✅</td></tr>`;
      notes.innerHTML = `<span class="text-green-700 font-semibold">Все значения в норме.</span><br>
                         Рекомендации: продолжать текущую стратегию.`;
    } else {
      data.anomalies.forEach(a => {
        tbody.innerHTML += `
          <tr>
            <td class="border px-2 py-1">${a.month}</td>
            <td class="border px-2 py-1">${a.cpl}</td>
            <td class="border px-2 py-1">${a.z_score.toFixed(2)}</td>
          </tr>
        `;
      });

      const z = data.anomalies[0].z_score;
      if (z > 0) {
        notes.innerHTML = `<span class="text-red-700 font-semibold">⚠ CPL выше нормы (Z = ${z.toFixed(2)}).</span><br>
                           Рекомендации: оптимизировать расходы и снизить стоимость заявок.`;
      } else {
        notes.innerHTML = `<span class="text-red-700 font-semibold">⚠ CPL ниже нормы (Z = ${z.toFixed(2)}).</span><br>
                           Рекомендации: проверить качество лидов и корректность данных.`;
      }
    }
  } catch (err) {
    console.error("Ошибка загрузки аномалий:", err);
  }
}

// === Init ===
async function init() {
  const metrics = await loadMetrics();
  await loadForecast(metrics);
  await loadAnomalies();

  setInterval(async () => {
    const metrics = await loadMetrics();
    await loadForecast(metrics);
    await loadAnomalies();
  }, 5000);
}

init();
