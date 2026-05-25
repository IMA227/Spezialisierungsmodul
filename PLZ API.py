#Loading libraries

import requests
import pandas as pd
from pathlib import Path


BASE_URL = "https://openplzapi.org/de"
OUTPUT_FILE = Path("....xlsx")
PAGE_SIZE = 50

HEADERS = {
    "accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def get_json(url, params=None):
    response = requests.get(url, headers=HEADERS, params=params, timeout=60)
    response.raise_for_status()
    return response


def fetch_federal_states():
    response = get_json(f"{BASE_URL}/FederalStates")
    data = response.json()

    if not isinstance(data, list):
        raise ValueError("Unexpected response format for federal states.")

    return data


def fetch_localities_for_state(state_code, state_name):
    url = f"{BASE_URL}/FederalStates/{state_code}/Localities"

    response = get_json(url, params={"page": 1, "pageSize": PAGE_SIZE})
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected response format for {state_name}, page 1.")

    total_pages = int(response.headers.get("x-total-pages", "1"))
    rows = []

    # keep only usable five-digit PLZ values
    def add_rows(items):
        for item in items:
            postal_code = str(
                item.get("postalCode", item.get("postalcode", ""))
            ).strip()

            bundesland = state_name or str(
                item.get("federalState", {}).get("name", "")
            ).strip()

            if len(postal_code) == 5 and postal_code.isdigit():
                rows.append({
                    "PLZ": postal_code,
                    "Bundesland": bundesland,
                })

    add_rows(data)

    for page in range(2, total_pages + 1):
        response = get_json(url, params={"page": page, "pageSize": PAGE_SIZE})
        data = response.json()

        if not isinstance(data, list):
            raise ValueError(f"Unexpected response format for {state_name}, page {page}.")

        add_rows(data)

    return rows


def main():
    all_rows = []

    for state in fetch_federal_states():
        state_code = str(state.get("key", "")).strip()
        state_name = str(state.get("name", "")).strip()

        if not state_code or not state_name:
            continue

        rows = fetch_localities_for_state(state_code, state_name)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No PLZ data retrieved.")

    df = pd.DataFrame(all_rows)

    
    df = (
        df
        .drop_duplicates(subset=["PLZ", "Bundesland"])
        .sort_values(["PLZ", "Bundesland"])
        .reset_index(drop=True)
    )

    df.to_excel(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()