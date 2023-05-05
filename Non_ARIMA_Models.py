# %%
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

import statsmodels.api as sm

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.regularizers import l2

# %%
df = pd.read_csv("public_data/processed_data.csv")

# %%
df.head()

# %%
df['datetime'] = pd.to_datetime(df['datetime'],format="%Y-%m-%d")
df = df[(df['datetime'].dt.year > 2008) & (df['datetime'].dt.year < 2023)]
df = df.set_index('datetime')
df = df.drop("Unnamed: 0", axis=1)
df['year'] = df.index.year
df['date'] = df.index.date

# %%
print("----- Descriptive Statistics -----")
print(df['calls'].describe())
print("Sum:", df['calls'].sum())

# %%
print("Fold Sizes")
print(pd.DataFrame(df.index.year.value_counts().sort_index()).rename(columns={'datetime': 'Fold Size'}).T)

# %%
year_group = df.groupby("year")['calls'].sum()
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(year_group, ".-", linewidth=4, markersize=15)
ax.set_xticks(year_group.index)
ax.set_xticklabels(["'"+str(y)[-2:] for y in year_group.index])
ax.grid(alpha=.4)
fmt = '{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
ax.set(title="Street and Sidewalk Cleaning Calls", xlabel="Year", ylabel="Annual Calls")
fig.tight_layout()
fig.savefig("figures/annual_trend.png", dpi=300)

# %%

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
groups = ['year', 'month', 'hourofday']
counter = 0
axes[0].set(ylabel="Calls")
for ax in axes:
    group = df.groupby(groups[counter])['calls'].sum()
    ax.plot(group, ".-", linewidth=4, markersize=15)
    ax.set_xticks(group.index)
    ax.set_xticklabels([str(y)[-2:] for y in group.index], fontsize=8)
    ax.grid(alpha=.4)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    ax.set( xlabel=groups[counter].title(), title=groups[counter].title())
    counter += 1
fig.tight_layout()
fig.savefig("figures/seasonality.png", dpi=300)

# %%
def evaluate_fit(test_preds, actual_calls):
    rmse = np.mean((actual_calls - test_preds)**2)**.5
    r2 = r2_score(actual_calls, test_preds)
    max_resid = np.max(np.abs(actual_calls - test_preds))
    print("rmse", rmse)
    print("r2", r2)
    print("max resid", max_resid)
    return rmse, r2, max_resid


# %%
n_steps = 48
X_ = df.copy()[['calls']]
for i in range(1, n_steps+1):
    X_[f'lag_{i}'] = X_['calls'].shift(i)

# %%
X_ = X_.dropna()

# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

allpreds = []
print("----- Naive Chain -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    lags = [x for x in np.ravel(y_train_scaled)[-48:]]
    preds = []
    for i in range(X_test_scaled.shape[0]):
        fitting = np.array(lags[-48:]).reshape(1, -1)
        first_pred = fitting.flatten()[-2]
        lags.append(first_pred)
        preds.append(first_pred)
    test_preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))

# %%
test_data['preds'] = test_preds

# %%
test_data.resample("1W")[['calls', 'preds']].sum().plot()

# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

allpreds = []
print("----- Regression Chain -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    lags = [x for x in np.ravel(y_train_scaled)[-48:]]
    preds = []
    for j in range(X_test_scaled.shape[0]):
        # Take the last 48 hours
        fitting = np.array(lags[-48:]).reshape(1, -1)
        # Predict given those values
        first_pred = lm.predict(fitting)
        first_pred = first_pred.flatten()[0]
        # add those predictions to the history
        lags.append(first_pred)
        preds.append(first_pred)
    
    test_preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    test_preds = test_preds
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds,
                                       actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))

# %%
train_data['predictions'] = scaler_y.inverse_transform(lm.predict(X_train_scaled))
test_data['predictions'] = test_preds

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
train_data.resample("1H")[['calls', 'predictions']].sum().plot(ax=axes[0])
axes[0].set(title="Train Data", ylabel="Calls")
test_data.resample("1H")[['calls', 'predictions']].sum().plot(ax=axes[1])
axes[1].set(title="Test Data")
fig.tight_layout()
fig.savefig("figures/Regression_Chain.png", dpi=300)

# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

allpreds = []
print("----- RF Chain -----")
feedback = input("Warning, this takes 45 minutes, press any key to continue")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    rf = RandomForestRegressor(verbose=0, n_jobs=-1, bootstrap=True)
    rf.fit(X_train_scaled, y_train_scaled)

    lags = [x for x in np.ravel(y_train_scaled)[-48:]]
    preds = []
    for i in range(X_test_scaled.shape[0]):
        # Take the last 48 hours
        fitting = np.array(lags[-48:]).reshape(1, -1)
        # Predict given those values
        first_pred = rf.predict(fitting)
        first_pred = first_pred.flatten()[0]
        # add those predictions to the history
        lags.append(first_pred)
        preds.append(first_pred)
    
    test_preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))



# %%
all_years = df.index.year.unique()
print("----- Naive Baseline Model -----")
rmses = []
r2s = []
max_resids = []
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = df[df.index.year == all_years[i]]
    test_data = df[df.index.year == all_years[i+1]]
    lag = 24*2
    test_preds = train_data['calls'].shift(lag).dropna()
    test_preds.index = test_preds.index + pd.DateOffset(years=1)
    actual_calls = test_data['calls'].iloc[lag:]
    merged = pd.merge(test_preds, actual_calls, left_index=True, right_index=True)
    actual_calls = merged['calls_y']
    test_preds = merged['calls_x']
    rmse, r2, max_resid = evaluate_fit(actual_calls, test_preds)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)
print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))
print()

# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

print("----- Regression With Lags -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
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
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))

# %%
split = int(X_.shape[0] * .8)
train_data = X_.iloc[:split, :]
test_data = X_.iloc[split:, :]

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
train_preds = scaler_y.inverse_transform(lm.predict(X_train_scaled))
actual_calls = y_test

# %%
X_temp = X_.copy()
X_temp['pred' ] = np.vstack([train_preds, test_preds])

# %%
fig, ax = plt.subplots()
agg_table = X_temp.resample("1M")[['calls', 'pred']].sum()
agg_table.columns = ["Actual Calls", "Predicted Calls"]
agg_table.plot(alpha=.7, ax=ax)
ax.set(xlabel="Day", ylabel="Calls", title="OLS w/ 48hr Lags")
ax.axvline(X_.index[split], label="Train-Test Split", linestyle=":")
ax.grid(alpha=.4)
ax.legend()
fig.savefig("figures/OLS.png", dpi=300)



# %%
all_years = df.index.year.unique()

rmses = []
r2s = []
max_resids = []

print("----- Ridge Regression With Lags -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[(X_.index.year == all_years[i])]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    lm = RidgeCV(alphas=np.linspace(0.001, 10, 10), cv=TimeSeriesSplit(n_splits=4))
    lm.fit(X_train_scaled, y_train_scaled)
    print("alpha =", lm.alpha_)
    test_preds = scaler_y.inverse_transform(lm.predict(X_test_scaled).reshape(-1, 1))
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))


# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

print("----- Lasso Regression With Lags -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    lm = LassoCV(alphas=np.linspace(0.001, 1, 10), cv=TimeSeriesSplit(n_splits=4))
    lm.fit(X_train_scaled, y_train_scaled)
    print("alpha =", lm.alpha_)
    test_preds = scaler_y.inverse_transform(lm.predict(X_test_scaled).reshape(-1, 1))
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))


# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

print("----- Random Forrest Regression With Lags -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    param_grid = {
    'n_estimators': [50, 100, 150]
    }
    
    rf = RandomForestRegressor(verbose=0, n_jobs=-1, bootstrap=True)
    lm = GridSearchCV(rf, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=4))
    lm.fit(X_train_scaled, y_train_scaled)
    print(lm.best_params_)
    test_preds = scaler_y.inverse_transform(lm.predict(X_test_scaled).reshape(-1, 1))
    actual_calls = y_test
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))


# %%
importances = lm.feature_importances_
std = np.std([tree.feature_importances_ for tree in lm.estimators_], axis=0)
indices = np.argsort(importances)[::-1] + 1

# %%
indices + 1

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Feature importances")
ax.bar(range(X_train_scaled.shape[1]), importances[indices-1], yerr=std[indices-1], align="center")
ax.set_yticks(range(X_train_scaled.shape[1]), indices)
ax.set_ylim([-1, X_train_scaled.shape[1]])
fig.tight_layout()
fig.savefig("figures/RF_plot.png", dpi=300)


# %%
all_years = df.index.year.unique()
rmses = []
r2s = []
max_resids = []

print("----- Vanilla RNN With Lags -----")
for i in range(len(all_years)-2):
    print(all_years[i], "-->", all_years[i+1])
    train_data = X_[X_.index.year == all_years[i]]
    test_data = X_[X_.index.year == all_years[i+1]]

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

    model = Sequential()
    model.add(SimpleRNN(200, activation='relu', return_sequences=True, input_shape=(n_steps, 1)), )
    model.add(Dropout(0.8))
    model.add(SimpleRNN(units=150, activation='relu', return_sequences=True))
    model.add(Dropout(0.6))
    model.add(SimpleRNN(units=100, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(SimpleRNN(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit the model to the training data
    model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
              y_train_scaled,
              batch_size=100,
              epochs=6)
    # make predictions for test data
    predictions = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))

    test_preds = scaler_y.inverse_transform(predictions)
    actual_calls = y_test
    print("Train")
    evaluate_fit(scaler_y.inverse_transform(model.predict(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1))),
                 y_train)
    print("Test")
    rmse, r2, max_resid = evaluate_fit(test_preds, actual_calls)
    rmses.append(rmse)
    r2s.append(r2)
    max_resids.append(max_resid)

print()
print("Mean RMSE:", np.mean(rmses))
print("Mean r2:", np.mean(r2s))
print("Max resid:", np.mean(max_resids))



