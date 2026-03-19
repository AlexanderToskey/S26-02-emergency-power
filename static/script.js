let map = L.map('map').setView([37.8, -78.5], 7);

// Base map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

let predictions = {};
let geojson;

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
    const data = predictions[fips];

    layer.on({

        mouseover: function () {
            let props;
            if (data && data.occurrence) {
                props = `
                    <b>${name} County</b><br/>
                    Outage Predicted<br/>
                    Projected # of Affected Customers: ${data.scope}<br/>
                    Projected Outage Duration: ${data.duration} hrs
                `;
            } else {
                props = `<b>${name} County</b><br/>No Outage Predicted`;
            }
            layer.bindTooltip(props).openTooltip();
            layer.setStyle({ weight: 3 });
        },

        mouseout: function () {
            layer.closeTooltip();
            geojson.resetStyle(layer);
        },

        click: function () {
            showInfoPanel(name, data);
        }
    });
}

// Displays the SHAP feature importance on the left side of the screen
function showInfoPanel(name, data) {
    const panel = document.getElementById("infoPanel");
    const mapDiv = document.getElementById("map");
    const title = document.getElementById("countyName");
    const details = document.getElementById("countyDetails");

    title.textContent = `${name} County`;

    if (data && data.occurrence) {
        details.innerHTML = `
            <b>Scope:</b> ${data.scope}<br/>
            <b>Duration:</b> ${data.duration} hrs<br/><br/>
            include feature justification here<br/><br/>

        `;
    } else {
        details.innerHTML = `
            No outage predicted.<br/><br/>
            include feature justification here<br/><br/>
        `;
    }

    panel.classList.remove("hidden");
    panel.classList.add("visible");

    setTimeout(() => {
        map.invalidateSize();
    }, 300);
}

// Populate the left panel with outage predictions
function populateSidebar(countiesData) {

    const list = document.getElementById("outageList");
    list.innerHTML = "";

    // Build array with all counties
    const allCounties = countiesData.features.map(feature => {
        const fips = String(feature.properties.GEOID);
        const name = feature.properties.NAME;
        const data = predictions[fips];

        return {
            fips,
            name,
            data,
            hasOutage: data && data.occurrence
        };
    });

    // Sort → outages first, then alphabetical
    allCounties.sort((a, b) => {

        // Outages first
        if (a.hasOutage && !b.hasOutage) return -1;
        if (!a.hasOutage && b.hasOutage) return 1;

        // Alphabetical within groups
        return a.name.localeCompare(b.name);
    });

    // Render all counties
    allCounties.forEach(({ fips, name, data, hasOutage }) => {

        const row = document.createElement("tr");
        row.className = "outage-row";

        // If no outage is predicted, gray it out
        if (!hasOutage) {
            row.classList.add("no-outage");
        }

        // Determine severity (only for outages)
        let severity = "yellow";
        let label = "Minor";

        if (hasOutage) {
            if (data.scope > 5000) {
                severity = "red";
                label = "Severe";
            } 
            else if (data.scope > 1000) {
                severity = "orange";
                label = "Moderate";
            }
        }

        row.innerHTML = `
            <td>${name} County</td>
            <td>
                ${hasOutage 
                    ? `<span class="badge ${severity}">${label}</span>`
                    : `<span class="no-outage-text">None</span>`
                }
            </td>
        `;

        // Helper to find county layer
        function getLayer() {
            let found = null;
            geojson.eachLayer(layer => {
                if (layer.feature.properties.GEOID === fips) {
                    found = layer;
                }
            });
            return found;
        }

        // Hover highlight only if there's an outage)
        row.addEventListener("mouseenter", () => {
            //if (!hasOutage) return;

            const layer = getLayer();
            if (layer) {
                layer.setStyle({
                    weight: 4,
                    color: "blue"
                });
            }
        });

        row.addEventListener("mouseleave", () => {
            //if (!hasOutage) return;

            const layer = getLayer();
            if (layer) {
                geojson.resetStyle(layer);
            }
        });

        // Click behavior
        row.addEventListener("click", () => {

            const layer = getLayer();

            if (layer) {

                map.fitBounds(layer.getBounds(), { padding:[200,200] });

                if (hasOutage) {
                    layer.setStyle({
                        weight: 4,
                        color: "blue"
                    });
                }

                showInfoPanel(name, data);

                setTimeout(() => {
                    geojson.resetStyle(layer);
                }, 3000);
            }
        });

        list.appendChild(row);
    });
}

// Fetch data
Promise.all([
    fetch("/api/counties").then(r => r.json()),
    fetch("/api/predictions").then(r => r.json())
]).then(([countiesData, preds]) => {
    // Get the outage predictions
    predictions = preds;

    // Extract only the Virginia counties
    const virginiaCounties = {
        type: "FeatureCollection",
        features: countiesData.features.filter(f => f.properties.STATEFP === "51")
    };

    // Add each county to the map
    geojson = L.geoJSON(virginiaCounties, {
        style: styleFunction,
        onEachFeature: onEachFeature
    }).addTo(map);

    // Bring all predicted counties to front
    geojson.eachLayer(layer => {
        const fips = String(layer.feature.properties.GEOID);
        const data = predictions[fips];
        if (data && data.occurrence) {
            layer.bringToFront();
        }
    });
    
    populateSidebar(virginiaCounties);

}).catch(err => console.error(err));

document.getElementById("closePanel").addEventListener("click", () => {
    const panel = document.getElementById("infoPanel");
    const mapDiv = document.getElementById("map");

    panel.classList.remove("visible");

    setTimeout(() => {
        map.invalidateSize();
    }, 300);
});