import pandas as pd
from sodapy import Socrata
import os

with open("secret.txt") as f:
    token = f.read().split("\n")[0]
client = Socrata("data.sfgov.org",
                 app_token=token)



def bulk_download_311(category: str = 'Street and Sidewalk Cleaning'):
    """Retrieve all 311 data for some category.
    This will download a .csv of all calls and save it
    locally. If the data has previously been retrieved,
    this will instead load that data locally.

    Args:
        category (str, optional): 311 category. Defaults to 'Street and Sidewalk Cleaning'.
    """
    cat = category.replace(" ", "_")
    csv_file_path = f"data/{cat}_data.csv"
    if os.path.isfile(csv_file_path):
        print("Data available locally")
        df_cleaning = pd.read_csv(csv_file_path)
    else:
        base_url = "https://data.sfgov.org/api/"
        url_311 = "views/vw6y-z8j6/rows.csv?accessType=DOWNLOAD"
        url = base_url + url_311
        print("Reading data (this will take ~10 minutes)")
        df = pd.read_csv(url)
        print(f"Querying `Category == {category}`")
        df_cleaning = df[df['Category'] == category]
        df_cleaning = df_cleaning.dropna(axis=1)
        df_cleaning.to_csv(csv_file_path)
    return df_cleaning

def process_data():
    df = bulk_download_311()
    print("Processing data")
    df['datetime'] = pd.to_datetime(df['Opened'],
    format="%m/%d/%Y %I:%M:%S %p")
    df = df.set_index('datetime')
    df = df[df.index < "2023-04-05"].sort_index()
    df_hourly = df.resample('1H').count().iloc[:, 0:1]
    df_hourly.columns = ['calls']
    df_hourly['month'] = df_hourly.index.month
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    df_hourly['dayofmonth'] = df_hourly.index.day
    df_hourly['hourofday'] = df_hourly.index.hour
    df_hourly.reset_index().to_csv("data/processed_data.csv")
    return df_hourly

def api_query_311(query: str = "service_name='Street and Sidewalk Cleaning'",
                  limit: int = 1000):
    """Query the 311 Socrata API for calls up to the limit
    This is good for recent calls, bad for historical.

    Args:
        query (str, optional): The Socrata query of choice.
        Defaults to "service_name='Street and Sidewalk Cleaning'".

    Returns:
        _type_: a Pandas DataFrame without calculated columns
    """
    results = client.get("vw6y-z8j6",
                         select="*",
                         limit=limit)
    results_df = pd.DataFrame.from_records(results)
    results_df_filtered = results_df[
        [col for col in results_df.columns if not col.startswith(":@")]
    ]
    return results_df_filtered

