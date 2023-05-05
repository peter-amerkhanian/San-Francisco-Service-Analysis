#!/usr/bin/env python
# coding: utf-8

# In[206]:


from sodapy import Socrata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
import retrieval_311


# In[207]:


df = pd.read_csv('data/processed_data.csv')
df = df.drop(labels=df.keys()[0], axis=1)
df['datetime'] = pd.to_datetime(df['datetime'],format="%Y-%m-%d")
df = df[(df['datetime'].dt.year > 2008) & (df['datetime'].dt.year < 2023)]


# In[208]:


full_hourly_count = df.copy()
full_hourly_count.columns = ['Hours', 'CaseID', 'Month', 'Day', 'Day_Month', 'Hour']
full_hourly_count['Year'] = full_hourly_count['Hours'].dt.year
full_hourly_count


# In[209]:


# normalize the data by year
# Clipping outliers outside of 3 standard deviations for each year
# saving a list of scalers to inverse forecasted data

normalized_data_array = []
scaler_list = []
for year in np.unique(full_hourly_count['Year']):
    scaler = StandardScaler()
    curr_year_data = full_hourly_count[full_hourly_count['Year']==year]['CaseID'].values
    curr_std = curr_year_data.std()
    clipped_curr_year_data = np.array([x if np.abs(x) < 3*curr_std else 3*curr_std for x in curr_year_data])
    normalized_data = scaler.fit_transform(clipped_curr_year_data.reshape(-1,1)).reshape(-1)
    normalized_data_array.append(normalized_data)
    scaler_list.append(scaler)
    
scaler_dict = {'scaler': scaler_list}
full_hourly_count['CaseID_norm'] = np.hstack(normalized_data_array)
full_hourly_count


# ## ARIMA

# In[210]:


plot_acf(full_hourly_count['CaseID_norm'].values.astype(float), lags=168, alpha=0.05)
plt.savefig('figures/ARIMA_autocorrelation', dpi=300)
plot_pacf(full_hourly_count['CaseID_norm'].values.astype(float), method='ywm')
plt.savefig('figures/ARIMA_partial_autocorrelation', dpi=300)
plt.show();


# ### 2021 predict 2022 tune hyperparameters

# In[211]:


data_2021 = full_hourly_count[full_hourly_count['Year']==2021]
data_2021

data_2022 = full_hourly_count[full_hourly_count['Year']==2022]
data_2022


# In[216]:


def tune_arima_hyperparams(data, forecast_length, p, d):
    '''
    Inputs:
        data
        p - max order of autoregressive model to evaluate
        d - max degree of differencing to evaluate
    Outputs:
        aic_array - numpy array of maic of each model evaluated
        mse_array - numpy array of mse of each model evaluated
        optimal_order - tuple, values of p,d,q that resulted in lowest mse
        forecast - 
    '''
    aic_array = np.ones((p,d))
    mse_array = np.ones((p,d))
    order_array = []
    for i in range(p):
        for j in range(d):
            model = ARIMA(data, order=(i+1,j+1,0))
            model_fit = model.fit()
            aic_array[i][j] = model_fit.aic
            mse_array[i][j] = model_fit.mse
            order_array.append(model.order)
            print(model.order)
            print(model_fit.aic)
            print(model_fit.mse)
            print("------------")

    optimal_order = order_array[np.argmin(mse_array)]
    best_model = ARIMA(data, order=optimal_order)
    model_fit = best_model.fit()
    print(model_fit.summary())
    forecast = model_fit.predict(end=forecast_length)
    return aic_array, mse_array, optimal_order, forecast


# In[215]:


aic_array, mse_array, optimal_order, forecast = tune_arima_hyperparams(data=data_2021['CaseID_norm'].values,                                                                        forecast_length=len(data_2022)-1,                                                                        p=48,                                                                        d=2)


# In[257]:


curr_scaler = scaler_dict['scaler'][np.argmax(np.unique(full_hourly_count['Year'].values)==2021)]
forecast_unnorm = curr_scaler.inverse_transform(forecast.reshape(-1,1)).reshape(-1)


# In[258]:


plt.plot(data_2021['CaseID'].values, c='b')
plt.plot(forecast_unnorm, c='r')
plt.title("Forecast vs. actual, 2022")
plt.xlabel("Hours (2022)")
plt.ylabel("Call volume")
plt.savefig("figures/Forecast_2022", dpi=300)
plt.show();


# In[259]:


plt.plot(aic_array)
plt.plot([], c='y', label='AIC, d=2')
plt.plot([], c='b', label='AIC, d=1')
plt.xlabel("Number of lagged observations, p")
plt.ylabel("AIC")
plt.title("Tuning hyper parameters p and d")
plt.legend()
plt.savefig('figures/ARIMA_AIC', dpi=300)
plt.show();


# In[345]:


plt.plot(mse_array)
plt.plot([], c='y', label='MSE, d=2')
plt.plot([], c='b', label='MSE, d=1')
plt.xlabel("Number of lagged observations, p")
plt.ylabel("MSE")
plt.title("Tuning hyper parameters p and d")
plt.legend()
plt.savefig('figures/ARIMA_RMSE', dpi=300)
plt.show();


# In[261]:


rmse = np.sqrt(mean_squared_error(forecast_unnorm, data_2022['CaseID']))
print('Test RMSE: %.3f' % rmse)


# In[262]:


r2 = r2_score(forecast_unnorm, data_2022['CaseID'])
print('Test R^2: %.3f' % r2)


# In[263]:


np.max(np.abs((forecast_unnorm - data_2022['CaseID'].values)))


# In[264]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()


# In[265]:


residuals.plot(kind='kde')


# In[266]:


residuals.describe()


# ### Do it for all years

# In[285]:


# Tuning all years with normalization and outlier clipping
# This code takes 1-1.5 hours to run if I include hyperparameter tuning
# Commenting that out and just using the optimal order, because when I've run it, there's no difference across models

all_years = np.unique(full_hourly_count['Year'].values)
full_forecast = np.array([])
orders = []
rmse_array = []
r2_array = []
for i in range(len(all_years)-1):
    year = all_years[i]
    print(year)
    next_year = all_years[i+1]
    current_year_data = full_hourly_count[full_hourly_count['Year']==year]['CaseID_norm'].values
    
    next_year_length = len(full_hourly_count[full_hourly_count['Year']==next_year])
    next_year_data = full_hourly_count[full_hourly_count['Year']==next_year]['CaseID'].values
    
    model = ARIMA(current_year_data, order=optimal_order)
    model_fit = model.fit()
#     print(model_fit.summary())
    forecast = model_fit.predict(end=next_year_length-1)
    
    
#     aic_array, mse_array, optimal_order, forecast = tune_arima_hyperparams(data=current_year_data, forecast_length=next_year_length-1, p=24, d=2)

    
#     aic_array, mse_array, optimal_order, forecast = tune_arima_hyperparams(data=clipped_current_year_data, forecast_length=next_year_length-1, p=24, d=2)
    
    # Need to select right scaler
    
    curr_scaler = scaler_dict['scaler'][np.argmax(np.unique(full_hourly_count['Year'].values)==year)]
    forecast_unnorm = curr_scaler.inverse_transform(forecast.reshape(-1,1)).reshape(-1)
    rmse = np.sqrt(mean_squared_error(next_year_data, forecast_unnorm))
    print('Test RMSE: %.3f' % rmse)
    r2 = r2_score(next_year_data, forecast_unnorm)
    print('Test R^2: %.3f' % r2)
    orders.append(optimal_order)
    rmse_array.append(rmse)
    r2_array.append(r2)
    full_forecast = np.concatenate([full_forecast, forecast_unnorm])


# In[340]:


# Calculate and plot average RMSE across models 
plt.plot(rmse_array, label='RMSE')
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE by yearly ARIMA model")
average_rmse = np.mean(rmse_array)
plt.hlines(average_rmse, xmin=0, xmax=13, colors='r', label='Average RMSE: %.2f' % average_rmse)
plt.legend()
plt.savefig("figures/ARIMA_model_RMSE", dpi=300)
plt.show();


# In[341]:


# Calculate and plot average R^2 across models 
plt.plot(r2_array, label='R^2')
plt.xlabel("Model")
plt.ylabel("R^2")
plt.title("R^2 by yearly ARIMA model")
average_r2 = np.mean(r2_array)
plt.hlines(average_r2, xmin=0, xmax=13, colors='r', label='Average R^2: %.2f' % average_r2)
plt.legend()
plt.show();


# In[342]:


# Calculate average max residual across models
max_resid_array = []
count = len(full_hourly_count[full_hourly_count['Year']==2009])
for i in range(len(np.unique(full_hourly_count['Year']))-2):
    increment = len(full_hourly_count[full_hourly_count['Year']==np.unique(full_hourly_count['Year'])[i+1]])
    curr_data = full_hourly_count.iloc[count:count+increment]['CaseID'].values
    max_resid = np.max(np.abs(curr_data - full_forecast[count:count+increment]))
    max_resid_array.append(max_resid)
    count += increment
    
print(np.mean(max_resid_array))


# In[347]:


fig, ax = plt.subplots(1,2, figsize=(12, 8))
ax[0].plot(full_forecast)
ax[0].set_title("Forecasted call volume")
ax[1].plot(full_hourly_count['CaseID'].values)
ax[1].set_title("Actual call volume")
fig.supxlabel("Hours (2008-2022)")
fig.supylabel("Hourly call volume")
ax[0].set_ylim(0,450)
ax[1].set_ylim(0,450)
plt.tight_layout()
plt.savefig('figures/ARIMA_forecast_vs_actual', dpi=300)
plt.show();

