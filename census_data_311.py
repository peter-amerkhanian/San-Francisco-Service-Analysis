import pickle
from us import states
from census import Census
import geopandas as gpd
import requests
import json


with open('secret.txt', 'r') as f:
    api_key = f.read().split("\n")[1]
census_client = Census(api_key)

sf_fips = '075'


def get_census_data(fields, c=census_client):
    """Retrieves 2019 5-year ACS data for San Francisco TRACTS
    Args:
        fields (list): list of variables to retrieve
        c (Census): an instantiated Census object
    Returns:
        pd.DataFrame: your data
    """
    path = "census_variables_json/sf_df.p"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            sf_df = pickle.load(f)
    else:
        sf_census = c.acs5.state_county_tract(fields=fields,
                                              state_fips=states.CA.fips,
                                              county_fips=sf_fips,
                                              tract="*",
                                              year=2019)
        sf_df = pd.DataFrame(sf_census)
        sf_df["GEOID"] = sf_df["state"] + sf_df["county"] + sf_df["tract"]
        sf_df = sf_df.drop(["state", "county", "tract"], axis=1)
        with open(path, "wb") as f:
            pickle.dump(sf_df, f)
    return sf_df


def get_census_data_blocks(fields, c=census_client):
    """Retrieves 2019 5-year ACS data for San Francisco BLOCKGROUPS
    Args:
        fields (list): list of variables to retrieve
        c (Census): an instantiated Census object
    Returns:
        pd.DataFrame: your data
    """
    path = "census_variables_json/sf_df_blocks.p"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            sf_df = pickle.load(f)
    else:
        sf_census = c.acs5.state_county_blockgroup(fields=fields,
                                                   state_fips=states.CA.fips,
                                                   county_fips=sf_fips,
                                                   blockgroup="*",
                                                   year=2019)
        sf_df = pd.DataFrame(sf_census)
        sf_df["GEOID"] = sf_df["state"] + sf_df["county"] + sf_df["tract"]
        sf_df = sf_df.drop(["state", "county", "tract"], axis=1)
        with open(path, "wb") as f:
            pickle.dump(sf_df, f)
    return sf_df


def clean_variable_names(sf_merge, fields):
    """Clean census column names
    Args:
        sf_merge (pd.DataFrame): original dataframe
    Returns:
        pd.DataFrame: dataframe with clean column names
    """
    base_url = "https://api.census.gov/data/2019/acs/acs5/variables/"
    for col in sf_merge.columns:
        if col in fields:
            file_path = f"census_variables_json/{col}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    resp_json = json.load(f)
            else:
                resp = requests.get(base_url + f"{col}.json")
                resp_json = resp.json()
            new_label = (resp_json['label']
                         .replace(":!!", "_")
                         .replace("!!", "_")
                         .replace(":", "_")
                         .replace(" ", "_")
                         .replace("Estimate", "")
                         .strip("_")
                         .replace(",", "")
                         )
            with open(file_path, "w") as f:
                json.dump(resp_json, f)
            sf_merge[new_label] = sf_merge[col]
            sf_merge = sf_merge.drop(col, axis=1)
    return sf_merge

# Note: Urls are from https://www2.census.gov/geo/tiger/TIGER2022/TRACT/


def retrieve_sf_shape(url, path, fips):
    """Retrieve San Francisco County's shape data
    from the census website
    Args:
        url (str): the url of the shapefile
        path (str): the path where it will be saved
    Returns:
        gpd.GeoDataFrame: gpd dataframe
    """
    ca_tract = gpd.read_file(url)
    sf_county_farallon = ca_tract[ca_tract['COUNTYFP'] == fips]
    sf_county_farallon.to_file(path)
    return sf_county_farallon


def open_sf_shape(url, path):
    """Open the San Francisco gpd either
    locally or from the census website
    Args:
        url (str): the url of the shapefile
        path (str): the path where it will be saved and/or opened from
    Returns:
        gpd.GeoDataFrame: gpd dataframe
    """
    if os.path.exists(path):
        print("Opening local file...")
        sf_county_farallon = gpd.read_file(path)
        print("Done")
    else:
        print("Retreiving from", f"{url}")
        sf_county_farallon = retrieve_sf_shape(url, path, sf_fips)
        print("Done")
    return sf_county_farallon


def remove_water(buffer_tracts):
    path = "sf_shapes/coastline.shp"
    if os.path.exists(path):
        coastline = gpd.read_file(path)
    else:
        coastline_url = "https://www2.census.gov/geo/tiger/TIGER2022/AREAWATER/tl_2022_06075_areawater.zip"
        coastline = gpd.read_file(coastline_url)
        coastline.crs = "epsg:4326"
        coastline.to_file(path)
    land_populated_df = buffer_tracts.overlay(coastline, how='difference')
    return land_populated_df
