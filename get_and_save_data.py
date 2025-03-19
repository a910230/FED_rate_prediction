# get_and_save_data.py
from fredapi import Fred
import pandas as pd

def fetch_fred_series(series_dict, fred_api_key):
    daily_cols = ["2-Year Treasury Yield", "10-Year Treasury Yield", "WTI Crude Oil Price"]
    quarterly_cols = ["Real GDP"]

    print("Fetching FRED data...")
    fred = Fred(api_key=fred_api_key)
    data = pd.DataFrame()
    for series_id, name in series_dict.items():
        try:
            series = fred.get_series(series_id)
            # Resample daily data to monthly before making a date frame to avoid alignment issue
            if name in daily_cols:
                series = series.resample("MS").mean()
            elif name in quarterly_cols:
                series = series.interpolate(method="linear")
            data[name] = series
            print(f"Downloaded {name} ({series_id})")
        except Exception as e:
            print(f"Error downloading {name} ({series_id}): {e}")

    data = data.ffill().dropna()
    return data

def get_and_save_data(fred_api_key, output_file):
    fred_series = {
        "FEDFUNDS": "Federal Funds Rate",
        "PCEPILFE": "Core PCE",
        "PCEPI": "Overall PCE",
        "UNRATE": "Unemployment Rate",
        "GDPC1": "Real GDP",
        "M2SL": "M2 Money Supply",
        "DGS2": "2-Year Treasury Yield",
        "DGS10": "10-Year Treasury Yield",
        "DCOILWTICO": "WTI Crude Oil Price",
        "UMCSENT": "Consumer Sentiment",
        # "DTWEXBGS": "USD Index", # data starts only from Jan 2006
        "IRSTCB01JPM156N": "BoJ Immediate Rate"
    }
    fred_data = fetch_fred_series(fred_series, fred_api_key)
    fred_data.to_csv(output_file)
    print(f"Data saved to {output_file}")