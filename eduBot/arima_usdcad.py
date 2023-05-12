import yfinance as yf
print("runnning")
def getYFinanceData(ticker, period):
    tickerObj = yf.Ticker(ticker)
    allData = tickerObj.history(period=period)
    # allData = tickerObj.history(period=period, proxy = "PROXY_SERVER")
    return allData


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cleanYFinanceData(allData):
    rate = pd.DataFrame(allData["Close"])
    rate = rate.reset_index()
    rate["Date"] = rate["Date"].apply(lambda t: t.strftime('%Y-%m-%d'))
    rate["Date"] = pd.to_datetime(rate["Date"])
    rate.set_index("Date", inplace=True)

    return rate


# Perform linear imputation on time series data

def imputeTimeSeries(rate):
    if (rate["Close"].isna().sum() == 0):
        return rate

    else:
        rate["Close"] = rate["Close"].interpolate(method='linear', inplace=True, limit_direction="both")
        return rate


# MAIN DRIVER CODE

asset = "USDCAD=X"
period = "3mo"

# CHANGE PERIOD HERE - 1year
allData = getYFinanceData(asset, period)
allData = allData.iloc[:-1, :]

rate = cleanYFinanceData(allData)
rate = imputeTimeSeries(rate)


"""
Part 1: Data Transformation
"""

from statsmodels.tsa.stattools import adfuller


# Null hypothesis: Time series data is not stationary
# Alternative hypothesis: Time series data is stationary

def isStationary(rate):
    if (rate.columns.str.contains("Difference").sum() == 0):
        rate["Difference"] = rate["Close"]

    results = adfuller(rate["Difference"])

    # check p-value is less than 0.05
    if results[1] <= 0.05:
        return True

    else:
        return False


def isStationaryExplain(rate):
    if (rate.columns.str.contains("Difference").sum() == 0):
        rate["Difference"] = rate["Close"]

    results = adfuller(rate["Difference"])
    resultsLabels = ['Test Statistic for ADF', 'p-value', 'Number of Lagged Observations Used',
                     'Number of Observations Used']

    for value, label in zip(results, resultsLabels):
        #print(label + ' : ' + str(value))
        pass
    # check p-value is less than 0.05
    if results[1] <= 0.05:
        #print("According to the ADF test, the p-value is less than 0.05. Therefore we reject the null hypothesis. We conclude the data is stationary")
        pass
    else:
        #print("According to the ADF test, the p-value is greater than 0.05. Therefore, we fail to reject the null hypothesis. We conclude the data is non-stationary ")
        pass

# Calculate the n-th degree difference of the time series data

def difference(rate, degree):
    testRate = rate.copy()

    if (testRate.columns.str.contains("Difference").sum() == 0):
        testRate["Difference"] = testRate["Close"]

    for i in range(degree):
        testRate["Difference"] = testRate["Difference"] - testRate["Difference"].shift(1)

    testRate = testRate.dropna()
    return testRate


# Calculate the degrees of transformation needed to make time series stationary

def degreeOfTransformation(rate):
    testRate = rate.copy()
    n = 0
    while (n < 5):
        if (isStationary(testRate)):
            return n

        else:
            testRate = difference(testRate, 1)
            n += 1

    return n


# WARNING: Only run once

def transformToStationary(rate):
    degree = degreeOfTransformation(rate)
    # print(degree)

    rate = difference(rate, degree)

    rate = rate.drop("Close", axis=1)
    rate = rate.rename(columns={"Difference": "Close"})

    return rate


def plotTransformation(rate):
    rateStationary = transformToStationary(rate)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].plot(rate)
    axs[1].plot(rateStationary)

    #plt.show()


# DRIVER CODE
#plotTransformation(rate)

"""
[AUTOMATIC METHOD] Part 2: Parameter Selection
"""
import pmdarima as pm
from pmdarima import model_selection
from pmdarima.arima import auto_arima

# MAIN DRIVER CODE pt.2
# DO NOT CHANGE PERIOD FROM MAX, use the first instance of getYFinanceData() to change period of stock
allDataMax = getYFinanceData(asset, "max")
rateAll = cleanYFinanceData(allDataMax)
rateAll = imputeTimeSeries(rateAll)
"""**Model Fitting**

[MANUAL METHOD] 
Part 2 : Parameter Selection
"""

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.base.tsa_model import ValueWarning
import warnings

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def plotACF(rate):
    rateStationary = transformToStationary(rate)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #sm.graphics.tsa.plot_acf(rateStationary['Close'], ax=ax, lags=100)


def plotPACF(rate):
    rateStationary = transformToStationary(rate)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #sm.graphics.tsa.plot_pacf(rateStationary['Close'], ax=ax, lags=100)


def plotParameterTests(rate):
    rateStationary = transformToStationary(rate)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sm.graphics.tsa.plot_acf(rateStationary['Close'], ax=axs[0], lags=100)
    #sm.graphics.tsa.plot_pacf(rateStationary['Close'], ax=axs[1], lags=100)


# DRIVER CODE
# plotParameterTests(rate)

# REMARK:
# We observe an AR(1) / MA(1) pattern

from sklearn.model_selection import train_test_split


# Splitting data into test and training
def getStartTrainData(rate, size=0.5):
    train, test = train_test_split(rate, shuffle=False, stratify=None, test_size=1 - size)
    return train


def getStartTestData(rate, size=0.5):
    train, test = train_test_split(rate, shuffle=False, stratify=None, test_size=size)
    return test


# Transfers the top datapoint from the testing dataset to become the bottom value in the training dataset
def transferTrainTestData(train, test):
    if len(test) == 0:
        return train, test

    datapoint = test.iloc[[0]]
    train = pd.concat([train, datapoint])
    test = test.iloc[1:]
    return train, test


def fitARIMAManual(rate, pTerm, degree, qTerm, train):
    degree = degreeOfTransformation(rate)

    arimaObj = sm.tsa.arima.ARIMA(train['Close'], order=(pTerm, degree, qTerm))

    modelARIMAManual = arimaObj.fit()
    return modelARIMAManual


def getMaxForesight(modelARIMAManual):
    maxForesight = max(int(modelARIMAManual.model_orders["ar"]), int(modelARIMAManual.model_orders["ma"]))
    return maxForesight


def getARIMAManualSummary(modelARIMAManual):
    return modelARIMAManual.summary()


def predictARIMAManual(modelARIMAManual, train):
    predictionARIMAManual = modelARIMAManual.predict(start=len(train),
                                                     end=len(train) - 1 + getMaxForesight(modelARIMAManual),
                                                     typ='levels', dynamic=True)
    # predictionARIMAManual = modelARIMAManual.forecast(steps = 1)
    return predictionARIMAManual


def getARIMAParameters(rateAll):
    autoARIMA = auto_arima(rateAll, start_p=0, start_q=0, max_p=10, max_q=10, m=0,
                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)  # set to stepwise

    pTerm = autoARIMA.order[0]
    degree = autoARIMA.order[1]
    qTerm = autoARIMA.order[2]

    params = []

    params.append(pTerm)
    params.append(degree)
    params.append(qTerm)

    return params


# MAIN DRIVER CODE
parameters = getARIMAParameters(rateAll)

if (parameters[0] + parameters[2]) < 1:
    raise Exception(
        "There are no autoregressive or moving average components to this data. This data is unsuitable for ARIMA analysis. Pick another dataset.")

modelARIMAManual = fitARIMAManual(rate, parameters[0], parameters[1], parameters[2], getStartTrainData(rate))
getARIMAManualSummary(modelARIMAManual)

from datetime import datetime as dt


def justify(a, invalid_val=0, axis=1, side='left'):
    """
    SOURCE: https://stackoverflow.com/questions/44558215/python-justifying-numpy-array/44559180#44559180

    Justifies a 2D array

    Parameters
    ----------
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """

    if invalid_val is np.nan:
        mask = ~np.isnan(a)
    else:
        mask = a != invalid_val
    justified_mask = np.sort(mask, axis=axis)
    if (side == 'up') | (side == 'left'):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(a.shape, invalid_val)
    if axis == 1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


# Return a dataframe of every day's predictions
# This function uses only actual values in its predictions (no predictions based off of predictions)
# It transfers a datapoint from the test dataset to the training dataset after that day's prediction is complete
# If the highest degree of the ARIMA is 1, it will return predictions for every day
# If the highest degree of the ARIMA is 2, it will return forecasts one AND two days in the future etc....

def allDynamicARIMAPredictions(rate, modelARIMAManual, trainStarting, testStarting):
    trainPredict = trainStarting.copy()
    testPredict = testStarting.copy()
    predictions = pd.DataFrame()

    modelARIMAManualPredict = modelARIMAManual
    degree = degreeOfTransformation(rate)
    pTerm = modelARIMAManualPredict.model_orders["ar"]
    qTerm = modelARIMAManualPredict.model_orders["ma"]

    foresight = getMaxForesight(modelARIMAManual)

    while (True):
        # print(trainPredict)
        # print(testPredict)

        modelARIMAManualPredict = fitARIMAManual(rate, pTerm, degree, qTerm, trainPredict)

        prediction = predictARIMAManual(modelARIMAManualPredict, trainPredict)

        if (foresight == 1):
            prediction = pd.DataFrame(prediction, columns=['Predicted Close'])
            predictions = pd.concat([predictions, prediction])

        else:
            a = prediction.T
            predictions = pd.concat([predictions, a])

        if len(testPredict) == 0:
            break

        # print(testPredict.iloc[0])
        # print(prediction)
        # print("")

        trainPredict, testPredict = transferTrainTestData(trainPredict, testPredict)

        # print("Train length: ", len(trainPredict))
        # print("Test length: ", len(testPredict))
        trainPredict = pd.DataFrame(trainPredict)
        testPredict = pd.DataFrame(testPredict)

    if foresight == 1:
        predictions.index = pd.to_datetime(predictions.index, format="%Y-%m-%d")

    else:
        predictions = cleanAllARIMAPredictions(predictions)

    predictions.index.rename('Date', inplace=True)

    return predictions


def cleanAllARIMAPredictions(predictions):
    foresight = getMaxForesight(modelARIMAManual)

    predictions = predictions.T
    predictions = pd.DataFrame(justify(predictions.to_numpy(), invalid_val=np.nan),
                               index=predictions.index,
                               columns=predictions.columns)
    predictions = predictions.iloc[:, foresight - 1:foresight]
    predictions = predictions.rename(columns={"predicted_mean": "Predicted Close With Lag " + str(foresight)})

    predictions.reset_index(inplace=True)

    predictions['index'] = predictions['index'].apply(lambda x: str(x))
    predictions['index'] = predictions['index'].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))

    predictions.index = pd.to_datetime(predictions.index, format="%Y-%m-%d")

    predictions = predictions.rename(columns={predictions.columns[0]: "Date"})
    predictions.set_index("Date", inplace=True)

    predictions = predictions.dropna()

    return predictions


# DRIVER CODE
predictions = allDynamicARIMAPredictions(rate, modelARIMAManual, getStartTrainData(rate), getStartTestData(rate))
getStartTestData(rate)



def plotAllARIMAPredictions(predictionsData, testData=getStartTestData(rate)):
    fig, ax = plt.subplots()

    ax.plot(testData, label="Observed Close")
    ax.legend(loc="upper right")

    ax.plot(predictionsData, label="Predicted Close")

    ax.legend(loc="upper right")

    if len(testData) > 60:
        plt.xticks(testData.index[::len(testData) // 15])

    plt.xticks(rotation=45, ha="right")
    ax.grid(color='w', which='both', linestyle='-', linewidth=1)

   # plt.show()


# DRIVER CODE
#plotAllARIMAPredictions(predictions)


def mergePredictionsAndTest(predictions, testData=getStartTestData(rate)):
    predictionsAndTest = testData.merge(predictions, left_index=True, right_index=True)
    return predictionsAndTest


# DRIVER CODE
predictionsAndTest = mergePredictionsAndTest(predictions)

# predictionsAndTest.plot()

"""**Model Validation**"""

from scipy import stats


def plotAllARIMAResiduals(modelARIMAManual):
    residuals = pd.DataFrame(modelARIMAManual.resid)
    residuals = residuals[(np.abs(stats.zscore(residuals)) < 3).all(axis=1)]

    fig, axs = plt.subplots(1, 1)
    axs.plot(residuals, label="Residuals")
    axs.legend(loc="upper right")

    plt.xticks(rotation=45, ha="right")
    #plt.show()

    # density plot of residuals
    residuals.plot(label="KDE Plot", kind='kde', )
    #plt.show()

    # summary stats of residuals
    print(residuals.describe())


#plotAllARIMAResiduals(modelARIMAManual)

"""**Error and P&L Analysis**"""


def predictionErrors(predictionsAndTest):
    predictionWithErrors = predictionsAndTest.copy()
    predictionWithErrors["Error"] = pd.Series(predictionsAndTest.iloc[:, 1]) - pd.Series(predictionsAndTest.iloc[:, 0])
    return predictionWithErrors


def plotErrors(predictionWithErrors):
    plt.plot(predictionWithErrors['Error'])
    plt.title('Prediction Error')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.xticks(rotation=45, ha="right")

    #plt.show()


# DRIVER CODE

predictionWithErrors = predictionErrors(predictionsAndTest)

# DRIVER CODE

#plotErrors(predictionWithErrors)


# Calculate PnL

def calculatePnL(closePredictedErrors, rate):
    foresight = getMaxForesight(modelARIMAManual)
    PnL = closePredictedErrors.copy()
    PnL['Shift Close'] = PnL['Close'].shift(foresight)

    PnL["Predicted Next-LAG-Day Close Difference"] = pd.Series(PnL.iloc[:, 1]) - PnL["Shift Close"]

    # If 1, we predict the stock will rise and we open a long position on the stock
    # If -1, we predict the stock will close and we open a short position on the stock
    PnL["Predicted Next-LAG-Day Close Difference Direction of Change"] = PnL[
        "Predicted Next-LAG-Day Close Difference"].apply(lambda x: 1 if x > 0 else -1)

    PnL["Actual Next-LAG-Day Close Difference"] = PnL['Close'].diff(foresight)

    PnL['Daily Absolute PnL'] = PnL['Predicted Next-LAG-Day Close Difference Direction of Change'] * PnL[
        'Actual Next-LAG-Day Close Difference']

    PnL["Daily Percentage PnL"] = PnL['Daily Absolute PnL'] / PnL['Close'].shift(foresight)

    return PnL


PnL = calculatePnL(predictionsAndTest, rate)


def calculateTotalAbsolutePnL(PnL):
    return PnL['Daily Absolute PnL'].sum()


def calculateTotalPercentagePnL(PnL):
    PnLPercentage = PnL.copy()

    PnLPercentagePlusOne = PnLPercentage['Daily Percentage PnL'].apply(lambda x: x + 1)

    if (PnLPercentagePlusOne.prod() >= 1):
        return PnLPercentagePlusOne.prod()

    else:
        return PnLPercentagePlusOne.prod() - 1


"""**Profit and Loss Visualization**"""

allData.tz_localize(None)
allData.index = pd.to_datetime(allData.index)


def plotPL(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=data['Date'], y=data['Daily Absolute PnL'],
               c=['red' if x < 0 else 'green' for x in data['Daily Absolute PnL']])
    ax.set_title('Daily Absolute PnL')
    ax.set_xlabel('Date')
    ax.set_ylabel('PnL')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
   # plt.show()

    # ax.spines['left'].set_position(('data', 0))


# might be the same as above but better as it is scaled

def plotPLPercentage(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=data['Date'], y=data['Daily Percentage PnL'],
               c=['red' if x < 0 else 'green' for x in data['Daily Percentage PnL']])
    ax.set_title('Daily Percentage PnL')
    ax.set_xlabel('Date')
    ax.set_ylabel('PnL%')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
   # plt.show()


# DRIVER CODE

PnLPlot = PnL.reset_index()
plotPLPercentage(PnLPlot)


def getGreatestGains(n, PnL):
    highest = PnL.nlargest(n, 'Daily Absolute PnL')
    return highest


def getGreatestLosses(n, PnL):
    lowest = PnL.nsmallest(n, 'Daily Percentage PnL')
    return lowest


# DRIVER CODE
getGreatestGains(10, PnL)

# DRIVER CODE
getGreatestLosses(10, PnL)

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade mplfinance
import mplfinance as mpf



# run once below
colorPlotData = allData.tz_localize(None).copy()
colorPlotData = colorPlotData.join(pd.DataFrame(PnL["Daily Absolute PnL"]), how="left")


colorPlotData.index = pd.DatetimeIndex(allData.index)
df_filtered = colorPlotData.loc[colorPlotData['Daily Absolute PnL'].notnull()]

#mpf.plot(df_filtered, type='line')
# dates_df = pd.DataFrame(allData.Predicted Next-Day Close Difference Direction of Change)
y1values = df_filtered['Close'].values
y2value = df_filtered['Low'].min()

where_values = df_filtered['Daily Absolute PnL'].values > 0
where_values1 = df_filtered['Daily Absolute PnL'].values < 0

fb1 = dict(y1=y1values, y2=y2value, where=where_values, alpha=0.5, color='green')
fb2 = dict(y1=y1values, y2=y2value, where=where_values1, alpha=0.5, color='red')

#mpf.plot(df_filtered, tight_layout=True, type='line',
     #    fill_between=[fb1, fb2])


getStartTrainData(rate)

getStartTestData(rate)

"""**Dashboard Output**"""


def getForecastPrice(predictionsAndTest):
    return predictionsAndTest["Predicted Close"][-1]


def getForecastReccomendation(predictionsAndTest):
    percentageChange = 1 + (predictionsAndTest["Predicted Close"][-1] / predictionsAndTest["Close"][-1])
    maxForesight = getMaxForesight(modelARIMAManual)

    if percentageChange > 0:
        print("we issue a BUY rating for this currency pair. Our model predicts a gain of " + str(
            round(percentageChange, 4)) + "% in closing price after the next " + str(maxForesight) + " trading day(s).")

    if percentageChange <= 0:
        print("we issue a SELL rating for this currency pair. Our model predicts a loss of ",
              str(round(percentageChange, 4)),
              "% in closing price after the next " + str(maxForesight) + " trading day(s).")


def getBackTestResults(PnL):
    percentPnL = calculateTotalPercentagePnL(PnL)
    absolutePnL = calculateTotalAbsolutePnL(PnL)

    print(
        "If this trend-following strategy's reccomendations were followed every day for the entire period, your profit ratio (gross of trading fees and slippage) would be " + str(
            round(percentPnL, 4)) + ". Your absolute returns would be $" + str(round(absolutePnL, 4)) + ".")


getForecastReccomendation(predictionsAndTest)
getBackTestResults(PnL)

