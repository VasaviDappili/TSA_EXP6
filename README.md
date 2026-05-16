# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16/05/2026



### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScalerfrom sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("C:/Users/admin/Downloads/Chocolate Sales (2).csv")

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

data['Amount'] = data['Amount'].replace('[\$,]', '', regex=True).astype(float)

data.set_index('Date', inplace=True)

print(data.head())

print(data.info())

data_monthly = data['Amount'].resample('MS').sum()

print(data_monthly.head())

data_monthly.plot(figsize=(10,5))
plt.title('Monthly Chocolate Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly.values.reshape(-1,1)
    ).flatten(),
    index=data_monthly.index
)

scaled_data.plot(figsize=(10,5))
plt.title('Scaled Data Plot')
plt.show()

decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

decomposition.plot()
plt.show()

scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]

test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(
    steps=len(test_data)
)


ax = train_data.plot(figsize=(10,5))

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend([
    "train_data",
    "test_predictions_add",
    "test_data"
])

ax.set_title('Visual Evaluation')

plt.show()



rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

print("RMSE :", rmse)

print(
    "Standard Deviation and Mean :",
    np.sqrt(scaled_data.var()),
    scaled_data.mean()
)



final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

final_predictions = final_model.forecast(
    steps=12
)


ax = scaled_data.plot(figsize=(10,5))

final_predictions.plot(ax=ax)

ax.legend([
    "scaled_data",
    "final_predictions"
])

ax.set_xlabel('Months')

ax.set_ylabel('Scaled Sales')

ax.set_title('Future Chocolate Sales Prediction')

plt.show()

print(final_predictions)
```

### OUTPUT:
<img width="826" height="484" alt="image" src="https://github.com/user-attachments/assets/97259d70-b7ab-4b66-8ad5-9668328597df" />

#### DECOMPOSED DATA PLOT:
<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/51a71457-a055-43e4-b482-d68a83967a82" />

#### TEST_PREDICTION
<img width="826" height="484" alt="image" src="https://github.com/user-attachments/assets/ee2ab2f8-bdb5-4098-86d6-780e2df619c8" />

#### FINAL_PREDICTION
<img width="846" height="484" alt="image" src="https://github.com/user-attachments/assets/7b85b2dd-3683-44a3-8f73-77609fd87f2f" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
