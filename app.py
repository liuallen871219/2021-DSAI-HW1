import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def get_train(data):
    past=pd.read_csv(data,encoding="utf-8")

    
    return past

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
def optimizeSARIMA(parameters_list, d, D, s,train):
    """Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table
def plotSARIMA(series, model, n_steps):
    """Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future    
    """
    
    # adding model values
    data = pd.DataFrame({"actual":series.values})
    sarima=pd.DataFrame({"sarima":model.fittedvalues.values})
    data=pd.concat([data,sarima],axis=1)
    data["sarima"][:s+d] = np.NaN
    data.head()
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    predict=forecast
    forecast = data.sarima.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data["actual"][s+d:], data["sarima"][s+d:])

    plt.figure(figsize=(18, 9))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    return predict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    train=get_train(args.training)
    ps = range(2, 5)
    d=1 
    qs = range(2, 5)
    Ps = range(0, 2)
    D=1 
    Qs = range(0, 2)
    s = 7 # 

    # creating list with all the possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)  # 36

    result_table = optimizeSARIMA(parameters_list, d, D, s,train["淨尖峰供電能力(MW)"])
    p, q, P, Q = result_table.parameters[0]
    best_model_1=sm.tsa.statespace.SARIMAX(train["淨尖峰供電能力(MW)"], order=(p,d , q), 
                                            seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model_1.summary())
    predict_1=plotSARIMA(train["淨尖峰供電能力(MW)"], best_model_1, 90)
    result_table = optimizeSARIMA(parameters_list, d, D, s,train["尖峰負載(MW)"])
    p, q, P, Q = result_table.parameters[0]
    best_model_2=sm.tsa.statespace.SARIMAX(train["尖峰負載(MW)"], order=(p,d , q), 
                                            seasonal_order=(P, D, Q, s)).fit(disp=-1)
    predict_2=plotSARIMA(train["尖峰負載(MW)"], best_model_2, 90)
    predict=predict_1-predict_2
    date=[int(i) for i in range(20210323,20210330)]
    out=predict[50:57]
    out=[int(i) for i in out]
    output=pd.DataFrame([date,out],index=["date","operating_reserve(MW)"])
    output=output.transpose()
    output.astype("int32")
    output.to_csv("submission.csv",encoding="utf-8",index=False)