let map = L.map('map').setView([37.8, -78.5], 7);

// Base map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

let predictions = {};
let geojson;

function formatCountyName(name, fips) {
    const countyCode = parseInt(fips.slice(2));

    if (countyCode >= 500) {
        return `${name} City`;
    } else {
        return `${name} County`;
    }
}

// Style counties
function styleFunction(feature) {

    const fips = feature.properties.GEOID;
    const countyData = predictions[fips];

    if (!countyData || !countyData.occurrence) {
        return { color: "#999", weight: 1, fillOpacity: 0.3 };
    }

    if (countyData.scope > 5000)
        return { fillColor: "red", weight: 1, color: "black", fillOpacity: 0.7 };

    if (countyData.scope > 1000)
        return { fillColor: "orange", weight: 1, color: "black", fillOpacity: 0.6 };

    return { fillColor: "yellow", weight: 1, color: "black", fillOpacity: 0.6 };
}

// Tooltip behavior
function onEachFeature(feature, layer) {
    const fips = String(feature.properties.GEOID);
    const name = feature.properties.NAME;
    const displayName = formatCountyName(name, fips);
    const data = predictions[fips];

    layer.on({
        mouseover: function () {
            let props;

            if (data && data.occurrence) {
                props = `
                    <b>${displayName}</b><br/>
                    Outage Predicted<br/>
                    Projected # of Affected Customers: ${data.scope}<br/>
                    Projected Outage Duration: ${data.duration} hrs
                `;
            } else {
                props = `<b>${displayName}</b><br/>No Outage Predicted`;
            }

            layer.bindTooltip(props).openTooltip();
            layer.setStyle({ weight: 3 });
        },

        mouseout: function () {
            layer.closeTooltip();
            geojson.resetStyle(layer);
        },

        click: function () {
            showInfoPanel(name, fips, data);
        }
    });
}

// ── Explanation panel rendering ────────────────────────────────────────────────

// Human-readable labels for weather features
const WX_LABELS = {
    tmax_c:           "Max Temp",
    tmin_c:           "Min Temp",
    awnd_ms:          "Avg Wind",
    wsfg_ms:          "Wind Gusts",
    prcp_mm:          "Precipitation",
    snow_mm:          "Snowfall",
    snwd_mm:          "Snow Depth",
};

const WX_UNITS = {
    tmax_c: "°C", tmin_c: "°C",
    awnd_ms: "m/s", wsfg_ms: "m/s",
    prcp_mm: "mm", snow_mm: "mm", snwd_mm: "mm",
};

// Active weather condition flags with display labels
const FLAG_LABELS = {
    wt_thunder:       "⛈ Thunderstorm",
    wt_fog:           "🌫 Fog",
    wt_snow:          "❄ Snow",
    wt_freezing_rain: "🌨 Freezing Rain",
    wt_ice:           "🧊 Ice",
    wt_blowing_snow:  "💨 Blowing Snow",
    wt_rain:          "🌧 Rain",
    wt_drizzle:       "🌦 Drizzle",
};

// Human-readable names for model feature columns shown in SHAP chart
const FEATURE_LABELS = {
    fips_code:    "County (FIPS)",
    year: "Year", month: "Month", day: "Day", dayofweek: "Day of Week",
    prcp_mm:      "Precipitation",
    snow_mm:      "Snowfall",
    snwd_mm:      "Snow Depth",
    tmax_c:       "Max Temp",
    tmin_c:       "Min Temp",
    awnd_ms:      "Avg Wind Speed",
    wsfg_ms:      "Wind Gusts",
    wt_fog:       "Fog",
    wt_thunder:   "Thunderstorm",
    wt_snow:      "Snow Flag",
    wt_freezing_rain: "Freezing Rain",
    wt_ice:       "Ice",
    wt_blowing_snow:  "Blowing Snow",
    wt_drizzle:   "Drizzle",
    wt_rain:      "Rain",
    has_weather_event: "Severe Weather",
    max_magnitude:     "Event Magnitude",
    magnitude_missing: "Magnitude Missing",
};

function featureLabel(name) {
    return FEATURE_LABELS[name] || name.replace(/_/g, " ");
}

function renderExplainPanel(container, explainData) {
    const occ  = explainData.occurrence;
    const prob = Math.round((explainData.occ_prob || 0) * 100);
    const w    = explainData.weather || {};
    const shap = explainData.shap;

    // ── Occurrence probability bar ─────────────────────────────────────────
    const probColor = prob >= 70 ? "#dc2626" : prob >= 40 ? "#f97316" : "#16a34a";
    const probHtml = `
        <div class="panel-section">
            <div class="panel-section-title">Outage Probability</div>
            <div class="prob-bar-wrap">
                <div class="prob-bar" style="width:${prob}%; background:${probColor}"></div>
            </div>
            <div class="prob-label">${prob}% likelihood of outage today</div>
        </div>`;

    // ── Scope + duration (only if outage predicted) ────────────────────────
    const predHtml = occ ? `
        <div class="panel-section">
            <div class="panel-section-title">Predictions</div>
            <div class="pred-grid">
                <div class="pred-item">
                    <span class="pred-label">Customers Affected</span>
                    <span class="pred-value">${Math.round(explainData.scope).toLocaleString()}</span>
                </div>
                <div class="pred-item">
                    <span class="pred-label">Est. Duration</span>
                    <span class="pred-value">${explainData.duration} hrs</span>
                </div>
            </div>
        </div>` : "";

    // ── Weather inputs ─────────────────────────────────────────────────────
    const wxRows = Object.entries(WX_LABELS).map(([key, label]) => {
        const val = w[key];
        const display = val != null ? `${parseFloat(val).toFixed(1)} ${WX_UNITS[key]}` : "—";
        return `<div class="wx-item">
                    <span class="wx-label">${label}</span>
                    <span class="wx-value">${display}</span>
                </div>`;
    }).join("");

    const activeFlags = Object.entries(FLAG_LABELS)
        .filter(([key]) => w[key] === 1)
        .map(([, label]) => `<span class="wx-flag">${label}</span>`)
        .join("");

    const wxHtml = `
        <div class="panel-section">
            <div class="panel-section-title">Today's Weather Inputs</div>
            <div class="weather-grid">${wxRows}</div>
            ${activeFlags ? `<div class="wx-flags">${activeFlags}</div>` : ""}
        </div>`;

    // ── SHAP feature contributions ─────────────────────────────────────────
    let shapHtml = "";
    if (shap && shap.features && shap.features.length > 0) {
        const maxAbs = Math.max(...shap.features.map(f => Math.abs(f.shap)), 0.0001);

        const bars = shap.features.map(f => {
            const pct  = (Math.abs(f.shap) / maxAbs * 100).toFixed(1);
            const dir  = f.shap >= 0 ? "positive" : "negative";
            const sign = f.shap >= 0 ? "+" : "";
            return `
                <div class="shap-row">
                    <span class="shap-name" title="${f.name}">${featureLabel(f.name)}</span>
                    <div class="shap-bar-wrap">
                        <div class="shap-bar ${dir}" style="width:${pct}%"></div>
                    </div>
                    <span class="shap-val ${dir}">${sign}${f.shap.toFixed(3)}</span>
                </div>`;
        }).join("");

        shapHtml = `
            <div class="panel-section">
                <div class="panel-section-title">Model Drivers (SHAP)</div>
                <div class="shap-legend">
                    <span class="shap-pos-dot"></span><span>pushes toward outage</span>
                    &nbsp;&nbsp;
                    <span class="shap-neg-dot"></span><span>pushes away</span>
                </div>
                ${bars}
            </div>`;
    } else if (shap === null) {
        shapHtml = `<div class="panel-section"><em>SHAP explainer unavailable.</em></div>`;
    }

    container.innerHTML = probHtml + predHtml + wxHtml + shapHtml;
}

// Opens the right-side panel and fetches live explain data for the county
function showInfoPanel(name, fips, _data) {
    const panel   = document.getElementById("infoPanel");
    const title   = document.getElementById("countyName");
    const details = document.getElementById("countyDetails");
    title.textContent = formatCountyName(name, fips);

   
    details.innerHTML = `<div class="panel-loading"><div class="panel-spinner"></div>Loading analysis…</div>`;

    panel.classList.remove("hidden");
    panel.classList.add("visible");
    setTimeout(() => { map.invalidateSize(); }, 300);

    fetch(`/api/explain/${fips}`)
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(explainData => renderExplainPanel(details, explainData))
        .catch(() => {
            details.innerHTML = `<p style="color:#888">Could not load analysis for this county.</p>`;
        });
}

// ── Sidebar population ─────────────────────────────────────────────────────────

function populateSidebar(countiesData) {

    const list = document.getElementById("outageList");
    list.innerHTML = "";

    const allCounties = countiesData.features.map(feature => {
        const fips = String(feature.properties.GEOID);
        const name = feature.properties.NAME;
        const data = predictions[fips];
        return { fips, name, data, hasOutage: data && data.occurrence };
    });

    allCounties.sort((a, b) => {
        if (a.hasOutage && !b.hasOutage) return -1;
        if (!a.hasOutage && b.hasOutage) return 1;
        return a.name.localeCompare(b.name);
    });

    allCounties.forEach(({ fips, name, data, hasOutage }) => {

        const row = document.createElement("tr");
        row.className = "outage-row";
        if (!hasOutage) row.classList.add("no-outage");

        let severity = "yellow", label = "Minor";
        if (hasOutage) {
            if (data.scope > 5000)      { severity = "red";    label = "Severe"; }
            else if (data.scope > 1000) { severity = "orange"; label = "Moderate"; }
        }

        row.innerHTML = `
            <td>${formatCountyName(name, fips)}</td>
            <td>${hasOutage
                ? `<span class="badge ${severity}">${label}</span>`
                : `<span class="no-outage-text">None</span>`
            }</td>`;

        function getLayer() {
            let found = null;
            geojson.eachLayer(layer => {
                if (layer.feature.properties.GEOID === fips) found = layer;
            });
            return found;
        }

        row.addEventListener("mouseenter", () => {
            const layer = getLayer();
            if (layer) layer.setStyle({ weight: 4, color: "blue" });
        });

        row.addEventListener("mouseleave", () => {
            const layer = getLayer();
            if (layer) geojson.resetStyle(layer);
        });

        row.addEventListener("click", () => {
            const layer = getLayer();
            if (layer) {
                map.fitBounds(layer.getBounds(), { padding: [200, 200] });
                if (hasOutage) layer.setStyle({ weight: 4, color: "blue" });
                showInfoPanel(name, fips, data);
                setTimeout(() => geojson.resetStyle(layer), 3000);
            }
        });

        list.appendChild(row);
    });
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────

Promise.all([
    fetch("/api/counties").then(r => r.json()),
    fetch("/api/predictions").then(r => r.json())
]).then(([countiesData, preds]) => {
    predictions = preds;

    const virginiaCounties = {
        type: "FeatureCollection",
        features: countiesData.features.filter(f => f.properties.STATEFP === "51")
    };

    geojson = L.geoJSON(virginiaCounties, {
        style: styleFunction,
        onEachFeature: onEachFeature
    }).addTo(map);

    geojson.eachLayer(layer => {
        const fips = String(layer.feature.properties.GEOID);
        if (predictions[fips] && predictions[fips].occurrence) layer.bringToFront();
    });

    populateSidebar(virginiaCounties);

}).catch(err => console.error(err));

document.getElementById("closePanel").addEventListener("click", () => {
    const panel = document.getElementById("infoPanel");
    panel.classList.remove("visible");
    setTimeout(() => { map.invalidateSize(); }, 300);
});
