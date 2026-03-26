import pandas as pd
import requests

# SpaceX API - publicly available
SPACEX_API = "https://api.spacexdata.com/v4/launches"

def fetch_launch_data():
    response = requests.get(SPACEX_API)
    launches = response.json()

    records = []
    for launch in launches:
        record = {
            "flight_number": launch.get("flight_number"),
            "date_utc": launch.get("date_utc"),
            "rocket": launch.get("rocket"),
            "success": launch.get("success"),
            "name": launch.get("name"),
            "details": launch.get("details"),
        }
        cores = launch.get("cores", [{}])
        if cores:
            core = cores[0]
            record["landing_success"] = core.get("landing_success")
            record["landing_type"] = core.get("landing_type")
            record["reused"] = core.get("reused")
            record["flights"] = core.get("flight")
            record["gridfins"] = core.get("gridfins")
            record["legs"] = core.get("legs")

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv("spacex_launches.csv", index=False)
    print(f"Saved {len(df)} records to spacex_launches.csv")
    return df

if __name__ == "__main__":
    fetch_launch_data()
