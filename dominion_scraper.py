# SUMMARY_URL = 'https://kubra.io/data/a92610c6-735b-47f8-ab7f-59b05d523fa2/public/summary-2/data.json'


import requests
import polyline
import itertools

STORMCENTER_ID = "9c691bb6-767e-4532-b00e-286ac9adc223"
VIEW_ID = "38b5394c-8bca-4dfd-ac59-b321615446bd"

def automated_va_scrape():



    state_url = f"https://kubra.io/stormcenter/api/v1/stormcenters/{STORMCENTER_ID}/views/{VIEW_ID}/currentState?preview=false"
    state_data = requests.get(state_url).json()
    
    interval_path = state_data['data']['interval_generation_data']
    interval_id = interval_path.split('/')[-1] 

    summary_url = f"https://kubra.io/{interval_path}/public/summary-2/data.json"

    print(f"Fetching system-wide summary from: {summary_url}")
    try:
        summary_resp = requests.get(summary_url)
        summary_data = summary_resp.json()
        # Extracting the val from: summaryFileData -> totals -> [0] -> total_cust_a -> val
        system_wide_total = summary_data['summaryFileData']['totals'][0]['total_cust_a']['val']
        print(f"System-wide Customers Affected (Official): {system_wide_total}\n")
    except Exception as e:
        print(f"Could not fetch summary total: {e}")
        system_wide_total = "Unknown"

    prefixes = ["0320"] 
    suffixes = ["".join(p) for p in itertools.product("0123", repeat=4)]
    quadkeys = [p + s for p in prefixes for s in suffixes]

    print(f"Scouting {len(quadkeys)} tiles for Virginia outages...")
    
    # base_url = f"https://kubra.io/cluster-data/201/072d01db-d26d-446b-83be-81f4fb94201c/{interval_id}/public/cluster-1/"
    
    total_customers = 0
    all_outages = []

    # print('base_url: ', base_url)

    for qk in quadkeys:
        qkh = qk[-3:][::-1]

        url = f"https://kubra.io/cluster-data/{qkh}/072d01db-d26d-446b-83be-81f4fb94201c/{interval_id}/public/cluster-1/{qk}.json"
        resp = requests.get(url)
        
        if resp.status_code == 200:
            data = resp.json()
            for item in data['file_data']:
                cust = item['desc']['cust_a']['val']
                geom = item['geom']['p'][0]
                coords = polyline.decode(geom)[0]
                
                total_customers += cust
                all_outages.append({"loc": coords, "cust": cust})
                print(f"[FOUND] Tile {qk}: {cust} customers out at {coords}")

    print(f"\nScrape Complete. Total Customers Affected: {total_customers}")


    return all_outages

if __name__ == "__main__":
    automated_va_scrape()