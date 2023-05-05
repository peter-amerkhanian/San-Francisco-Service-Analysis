# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import retrieval_scripts.retrieval_311
import census_data_311
import geopandas as gpd
import os

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# %%
df = retrieval_311.bulk_download_311()

# %%
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# %%
# variables: https://api.census.gov/data/2019/acs/acs5/variables.html
hispanic = [f'B03002_0{i:02d}E' for i in range(1, 13)]
other_vars = ['B01003_001E',
              'B19013_001E', # median income
              'B01002_001E', # 
              'B25024_001E', # total structures
              'B25024_002E' # single family homes
              ]
fields = hispanic + other_vars
census_df = census_data_311.get_census_data(fields)
sf_df = census_data_311.clean_variable_names(census_df, fields)

# %%
shape_url = "https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_06_tract.zip"
path = "sf_shapes/sf_county_farallon.shp"

sf_county_farallon = census_data_311.open_sf_shape(shape_url, path)

# %%
sf_merge = sf_county_farallon.merge(sf_df, on = "GEOID", how="inner")

# %%
sf_merge = sf_merge[sf_merge['Total'] >= 1]
gdf = gdf[gdf['Latitude'] != 0]

# %%
sf_merge.crs = "epsg:4326"
gdf.crs = "epsg:4326"

# %%
land_populated_df = census_data_311.remove_water(sf_merge)

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title("311 Street Cleaning Calls, 2016-2023")
land_populated_df.exterior.plot(linewidth=.6, color='grey', alpha=.8, ax=ax)
gdf.sample(10000).plot(ax=ax, color="tab:red", marker=".", alpha=.25, linewidth=0)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("figures/311_map.png", dpi=300)

# %%
for col in land_populated_df.columns:
    if col.startswith("Total_Not_Hispanic_or_Latino"):
        land_populated_df[col+"%"] = land_populated_df[col] / land_populated_df['Total_Not_Hispanic_or_Latino']
    elif col.startswith("Total_Hispanic_or_Latino"):
        land_populated_df[col+"%"] = land_populated_df[col] / land_populated_df['Total']


# %%
jointracts = gpd.sjoin(left_df=gdf,
                       right_df=land_populated_df,
                       how='left')


# %%
jointracts['datetime'] = pd.to_datetime(jointracts['Opened'],format="%m/%d/%Y %I:%M:%S %p")

# %%
jointracts = jointracts[(jointracts['datetime'].dt.year > 2008) & (jointracts['datetime'].dt.year < 2023)]

# %%
jointracts['date'] = jointracts['datetime'].dt.date
jointracts['hour'] = jointracts['datetime'].dt.hour
jointracts['year'] = jointracts['datetime'].dt.year

# %%
grouped_df = jointracts.groupby(['GEOID', 'datetime']).size().reset_index()

# %%
grouped_df = grouped_df.rename(columns={0: 'calls'})
grouped_df = grouped_df.set_index("datetime")

# %%
n_steps = 48

# %%
def evaluate_fit(test_preds, actual_calls, verbose=True):
    rmse = np.mean((actual_calls - test_preds)**2)**.5
    r2 = r2_score(actual_calls, test_preds)
    max_resid = np.max(np.abs(actual_calls - test_preds))
    if verbose:
        print("rmse", rmse)
        print("r2", r2)
        print("max resid", max_resid)
    return rmse, r2, max_resid

# %%
tract_level_accuracy = {}

print("----- Regression With Lags -----")
for geoid in grouped_df['GEOID'].unique():
    tract_level_accuracy[geoid] = []
    # print(geoid)
    tract = grouped_df[grouped_df['GEOID'] == geoid]
    tract = tract.resample('1H')[['calls']].count()
    X_ = tract.copy()[['calls']]
    for i in range(1, n_steps+1):
        X_[f'lag_{i}'] = X_['calls'].shift(i)
    X_ = X_.dropna()
    all_years = X_.index.year.unique()

    rmses = []
    r2s = []
    max_resids = []
    for i in range(len(all_years)-2):
        # print(all_years[i], "-->", all_years[i+1])
        train_data = X_[X_.index.year == all_years[i]].diff(1).dropna()
        test_data = X_[X_.index.year == all_years[i+1]].diff(1).dropna()

        X_train = train_data.drop('calls', axis=1).values
        y_train = train_data[['calls']].values
        X_test = test_data.drop('calls', axis=1).values
        y_test = test_data[['calls']].values

        scaler_X = StandardScaler()
        scaler_X.fit(X_train)
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_test_scaled = np.clip(X_test_scaled, -3, 3)

        lm = LinearRegression()
        lm.fit(X_train_scaled, y_train_scaled)
        test_preds = scaler_y.inverse_transform(lm.predict(X_test_scaled))
        actual_calls = y_test
        rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls, verbose=False)
        rmses.append(rmse)
        r2s.append(r2)
        max_resids.append(max_resid)
    # print("Mean RMSE:", np.mean(rmses))
    # print("Mean r2:", np.mean(r2s))
    # print("Max resid:", np.mean(max_resids))
    tract_level_accuracy[geoid].append(np.mean(rmses))
    tract_level_accuracy[geoid].append(np.mean(r2s))
    tract_level_accuracy[geoid].append(np.mean(max_resids))

# %%
tract_performance = pd.DataFrame(tract_level_accuracy).T

# %%
tract_performance.columns = ["rmse", "r^2", "max_residual"]
tract_performance = tract_performance.reset_index()

# %%
performance_with_shape = sf_merge[['GEOID', 'geometry']].merge(tract_performance, left_on="GEOID", right_on="index")

# %%
performance_with_shape.to_csv("figures/OLS_tract_performance.csv")

# %%
print(performance_with_shape.describe())

# %%
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
count = 0
colors = ["Reds", "Greens", "Oranges"]
for measure in ["rmse", "r^2", "max_residual"]:
    performance_with_shape.plot(measure, legend=True, cmap=colors[count], ax=axes[count],)
    performance_with_shape.exterior.plot(linewidth=.3, color="black", ax=axes[count])
    axes[count].set_axis_off()
    axes[count].set_title("OLS: " + measure.title())
    count += 1
fig.tight_layout()
fig.savefig("figures/geospatial_performance_OLS.png", dpi=300)

# %%
tract_level_accuracy = {}
print("----- RF With Lags -----")

for geoid in grouped_df['GEOID'].unique():
    tract_level_accuracy[geoid] = []
    # print(geoid)
    tract = grouped_df[grouped_df['GEOID'] == geoid]
    tract = tract.resample('1H')[['calls']].count()
    X_ = tract.copy()[['calls']]
    for i in range(1, n_steps+1):
        X_[f'lag_{i}'] = X_['calls'].shift(i)
    X_ = X_.dropna()
    all_years = X_.index.year.unique()

    rmses = []
    r2s = []
    max_resids = []
    for i in range(len(all_years)-2):
        # print(all_years[i], "-->", all_years[i+1])
        train_data = X_[X_.index.year == all_years[i]].diff(1).dropna()
        test_data = X_[X_.index.year == all_years[i+1]].diff(1).dropna()

        X_train = train_data.drop('calls', axis=1).values
        y_train = train_data[['calls']].values
        X_test = test_data.drop('calls', axis=1).values
        y_test = test_data[['calls']].values

        scaler_X = StandardScaler()
        scaler_X.fit(X_train)
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_test_scaled = np.clip(X_test_scaled, -3, 3)

        y_train_scaled = y_train_scaled.flatten()
        y_test_scaled = y_test_scaled.flatten()

        lm = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        lm.fit(X_train_scaled, y_train_scaled)
        test_preds = scaler_y.inverse_transform(lm.predict(X_test_scaled).reshape(-1, 1))
        actual_calls = y_test
        rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls, verbose=False)
        rmses.append(rmse)
        r2s.append(r2)
        max_resids.append(max_resid)
    # print("Mean RMSE:", np.mean(rmses))
    # print("Mean r2:", np.mean(r2s))
    # print("Max resid:", np.mean(max_resids))
    tract_level_accuracy[geoid].append(np.mean(rmses))
    tract_level_accuracy[geoid].append(np.mean(r2s))
    tract_level_accuracy[geoid].append(np.mean(max_resids))

# %%
tract_performance = pd.DataFrame(tract_level_accuracy).T

# %%
tract_performance.columns = ["rmse", "r^2", "max_residual"]
tract_performance = tract_performance.reset_index()

# %%
performance_with_shape = sf_merge[['GEOID', 'geometry']].merge(tract_performance, left_on="GEOID", right_on="index")

# %%
performance_with_shape.to_csv("figures/RF_tract_performance.csv")

# %%
print(performance_with_shape.describe())

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
count = 0
colors = ["Reds", "Greens", "Oranges"]
for measure in ["rmse", "r^2", "max_residual"]:
    performance_with_shape.plot(measure, legend=True, cmap=colors[count], ax=axes[count],)
    performance_with_shape.exterior.plot(linewidth=.3, color="black", ax=axes[count])
    axes[count].set_axis_off()
    axes[count].set_title("RandomForest\n" + measure.title())
    count += 1
fig.tight_layout()
fig.savefig("figures/geospatial_performance_RF.png", dpi=300)


