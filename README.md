# San-Francisco-Service-Analysis

## Project Title: Forecasting and Inference for Public Sector service provisioning

**Team Members**: Peter Amerkhanian, Jared Schober

**Datasets**:
- SF 311 calls for service (2008-2023)
- Daily Weather (2015-2023)
- Census American Community Survey (2015-2023)
## WORKFLOW

```
git pull
# start working
git add .
git commit -m "your message here"
git push
```

## Project Description:
Forecast the need for public services – ~~fire, police,~~ and 311 calls for service – so that a city can more accurately prepare for service delivery. We will use the data as a time series and will aim to produce forecasts that are accurate at one week.  

Infer patterns in historical service allocation and quality. ~~We will predict the response time for each type of call for service~~, and examine how this varies with respect to various geographic and demographic covariates. We will also examine how service volume per-capita varies across locations.


## Methods:
- Linear Regression, trained on time series lags (likely at the hour), weather patterns (likely at the hour), and city economic conditions (likely at the year or month).
- ARIMA
- ~~Gradient Boosting, specifically extreme gradient boosting (XGBoost)~~
- Recurrent Neural Network

## Methodological questions:
- Will deep learning outperform simpler models like linear regression and ~~gradient boosted regression trees~~ for this task?
- Is deep learning necessary for achieving our goal? This question is motivated by a working paper by (Elsayed et. al. 2021)
- Can deep learning and ensemble methods be used for effective inference?

## Deliverables:
- Phase 1 (by April 15th): Data gathered from various sources into a single dataset, imputation of missing data, feature engineering, exploratory data analysis
- Phase 2 (by May 3rd): Forecasting models trained (regression, gradient boosting, neural network), inference conducted, writeup completed
