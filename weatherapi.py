import requests
import pandas as pd
from time import sleep

# =========================
# INSERT YOUR API KEY HERE
# =========================
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

# Census API endpoint for counties in Virginia (state FIPS = 51)
url = "https://api.census.gov/data/2020/dec/pl?get=NAME&for=county:*&in=state:51"

response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])

# Rename columns
df.rename(columns={
    "NAME": "county_name",
    "state": "state_fips",
    "county": "county_fips"
}, inplace=True)

# Create full FIPS (state + county)
df["fips"] = df["state_fips"] + df["county_fips"]


# Function to get lat/lon using Google Places API
def get_lat_lon(county_name):
    try:
        query = f"{county_name}, Virginia, USA"
        endpoint = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

        params = {
            "input": query,
            "inputtype": "textquery",
            "fields": "geometry",
            "key": GOOGLE_API_KEY
        }

        response = requests.get(endpoint, params=params)
        result = response.json()

        if result.get("candidates"):
            location = result["candidates"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
    except Exception as e:
        print(f"Error for {county_name}: {e}")

    return None, None


# Get coordinates
latitudes = []
longitudes = []

for county in df["county_name"]:
    lat, lon = get_lat_lon(county)
    latitudes.append(lat)
    longitudes.append(lon)
    sleep(0.2)  # Google allows faster requests than Nominatim

df["latitude"] = latitudes
df["longitude"] = longitudes

# Save to CSV
df.to_csv("virginia_geo.csv", index=False)

print("CSV file 'virginia_geo.csv' created successfully.")