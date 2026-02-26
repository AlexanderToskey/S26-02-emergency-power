let map = L.map('map').setView([37.8, -78.5], 7);  // Centered on VA

// Base map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

let predictions = {};
let geojson;

// Style function
function styleFunction(feature) {

    const fips = feature.properties.GEOID;
    const countyData = predictions[fips];

    if (!countyData || !countyData.occurrence) {
        return { color: "#999", weight: 1, fillOpacity: 0.3 };
    }

    // Color severity based on scope
    if (countyData.scope > 5000) return { fillColor: "red", weight: 1, color: "black", fillOpacity: 0.7 };
    if (countyData.scope > 1000) return { fillColor: "orange", weight: 1, color: "black", fillOpacity: 0.6 };
    return { fillColor: "yellow", fillOpacity: 0.5 };

    return {
        fillColor: "#9ecae1",
        weight: 1,
        opacity: 1,
        color: 'blue',
        fillOpacity: 0.5
    };
}

// Tooltip function
function onEachFeature(feature, layer) {
    const fips = String(feature.properties.GEOID);
    const name = feature.properties.NAME;
    const data = predictions[fips];

    layer.on({
        mouseover: function(e) {
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
        mouseout: function(e) {
            layer.closeTooltip();
            geojson.resetStyle(layer);
        }
    });
}

// Fetch counties and predictions, then render map
Promise.all([
    fetch("/api/counties").then(r => r.json()),
    fetch("/api/predictions").then(r => r.json())
]).then(([countiesData, preds]) => {

    predictions = preds;

    // Filter Virginia only
    const virginiaCounties = {
        type: "FeatureCollection",
        features: countiesData.features.filter(f => f.properties.STATEFP === "51")
    };

    geojson = L.geoJSON(virginiaCounties, {
        style: styleFunction,
        onEachFeature: onEachFeature
    }).addTo(map);
}).catch(err => console.error(err));