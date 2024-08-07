#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# Load the data from 'indices.csv', setting the 'date' column as the index. 
# Print the first 10 rows and statistical description of the data, and data structure info.
# visualizes the raw data.
"""

historical_data  = pd.read_csv("indices.csv", index_col = ["date"], parse_dates = True)
historical_data.plot(figsize=(10,12), subplots = True)

print(historical_data.head(10))
print(historical_data.describe())
print(historical_data.info())
# %%
"""
Data cleaning process
# - Filters the data starting from 1981-01-31 according to "TOPIX" valid data range.
# - Converts all values to their absolute values (removing negative signs in "MSCI UK") .
# - Forward fills (ffill) and backward fills (bfill) to handle missing values in "MSCI US".
# - Replaces zeros with NaN and then performs linear interpolation to estimate missing values keep the trend similarty.
"""
data_cleaned = historical_data.loc["1981-01-31":,:]
data_cleaned = data_cleaned.abs()
data_cleaned = data_cleaned.ffill().bfill()
data_cleaned  = data_cleaned.replace(0.0, np.nan).interpolate()

# Plot the cleaned data to visualize changes
data_cleaned.plot(figsize=(14, 7), subplots= True)  

#check the data info and statistical description of data after data cleaning
print(data_cleaned.describe())
print(data_cleaned.info())
# %%
# Calculate the 6-month momentum for each index ("MSCI UK", "MSCI US", "TOPIX")
data_cleaned['momentum_MSCI UK_6M'] = (data_cleaned['MSCI UK'] / data_cleaned['MSCI UK'].shift(6)) - 1
data_cleaned['momentum_MSCI US_6M'] = (data_cleaned['MSCI US'] / data_cleaned['MSCI US'].shift(6)) - 1
data_cleaned['momentum_TOPIX_6M'] = (data_cleaned['TOPIX'] / data_cleaned['TOPIX'].shift(6)) - 1

# Plot the momentum data for each index
data_cleaned[['momentum_MSCI UK_6M','momentum_MSCI US_6M','momentum_TOPIX_6M']].plot(figsize=(14, 7), subplots=True)  
#%%
# Determine the index with the highest momentum each month and calculate the returns based on the strategy that invests in the highest momentum index
data_cleaned['best_momentum_6M'] = data_cleaned[['momentum_MSCI UK_6M', 'momentum_MSCI UK_6M', 'momentum_TOPIX_6M']].idxmax(axis=1)
data_cleaned['momentum_strategy_returns_6M'] = np.where(data_cleaned['best_momentum_6M'] == 'momentum_MSCI UK_6M', data_cleaned['momentum_MSCI UK_6M'], 
                                            np.where(data_cleaned['best_momentum_6M'] == 'momentum_MSCI US_6M', data_cleaned['momentum_MSCI US_6M'], 
                                                    data_cleaned['momentum_TOPIX_6M']))
data_cleaned['cumulative_returns_momentum_6M'] = ((1 + data_cleaned['momentum_strategy_returns_6M']).cumprod()) - 1 
# %%

# Calculate the equal weight returns for comparison (simple average of the three indices)
data_cleaned['equal_weight_returns_6M'] = (data_cleaned['momentum_MSCI UK_6M'] + data_cleaned['momentum_MSCI US_6M'] + data_cleaned['momentum_TOPIX_6M']) / 3
data_cleaned['cumulative_returns_equal_6M'] = ((1 + data_cleaned['equal_weight_returns_6M']).cumprod())-1 
# %%

# Plot the cumulative returns for both strategies using a logarithmic scale to manage extreme values.
plt.figure(figsize=(14, 7))
plt.plot(data_cleaned['cumulative_returns_momentum_6M'], label='Momentum Strategy_6M', color='red')
plt.plot(data_cleaned['cumulative_returns_equal_6M'], label='Equal Weight Strategy_6M', color='blue')
plt.title('Cumulative Returns: Momentum vs. Equal Weight Strategy')
plt.xlabel('Year')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Use logarithmic scale to handle 'inf' values and get a clearer view
plt.show()

# %%
def calculate_momentum(data,  periods):
    """
    Calculate momentum and cumulative returns for different periods.
    
    Parameters:
        data (DataFrame): The input data frame containing the index prices.
        periods(list): A list of integers representing the periods in months.
    """
    for months in periods:
        # Calculate momentum in given periods
        data[f'momentum_MSCI UK_{months}M'] = (data['MSCI UK'] / data['MSCI UK'].shift(months))- 1
        data[f'momentum_MSCI US_{months}M'] = (data['MSCI US'] / data['MSCI US'].shift(months)) - 1
        data[f'momentum_TOPIX_{months}M'] = (data['TOPIX'] / data['TOPIX'].shift(months)) - 1

        # Determine the index with the highest momentum each month
        data[f'best_momentum_{months}M'] = data[[f'momentum_MSCI UK_{months}M', f'momentum_MSCI US_{months}M', f'momentum_TOPIX_{months}M']].idxmax(axis=1)
        
        # Calculate the returns based on the strategy that invests in the highest momentum index
        data[f'momentum_strategy_returns_{months}M'] = np.where(
            data[f'best_momentum_{months}M'] == f'momentum_MSCI UK_{months}M', data[f'momentum_MSCI UK_{months}M'],
            np.where(
                data[f'best_momentum_{months}M'] == f'momentum_MSCI US_{months}M', data[f'momentum_MSCI US_{months}M'], 
                data[f'momentum_TOPIX_{months}M']
            )
        )
        
        # Calculate cumulative returns for the momentum strategy
        data[f'cumulative_returns_momentum_{months}M'] = (1 + data[f'momentum_strategy_returns_{months}M']).cumprod() 

        # Calculate the equal weight returns for comparison
        data[f'equal_weight_returns_{months}M'] = (data[f'momentum_MSCI UK_{months}M'] + data[f'momentum_MSCI US_{months}M'] + data[f'momentum_TOPIX_{months}M']) / 3
        data[f'cumulative_returns_equal_{months}M'] = (1 + data[f'equal_weight_returns_{months}M']).cumprod() 

calculate_momentum(data_cleaned, [3, 6, 12])

# Plot the cumulative returns for both strategies
plt.figure(figsize=(14, 7))
colors_momentum = ['#FF9999', '#FF6666', '#FF0000']  # Light to dark red
colors_equal = ['#99FF99', '#66FF66', '#00FF00']  # Light to dark green
i = 0
for months in [3, 6, 12]:
    plt.plot(data_cleaned.index, data_cleaned[f'cumulative_returns_momentum_{months}M'], label=f'Momentum {months}M', color=colors_momentum [i])
    plt.plot(data_cleaned.index, data_cleaned[f'cumulative_returns_equal_{months}M'], label=f'Equal Weight {months}M', color=colors_equal [i])
    i += 1

plt.title('Cumulative Returns: Momentum vs Equal Weight Strategies')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Cumulative Returns',fontsize=15)
plt.legend(fontsize=11)
plt.grid(True)
plt.yscale('log')   # Use logarithmic scale to handle 'inf' values and get a clearer view
plt.show()

# %%
"""
6-month periods momentum analysisby adding buy signal
    
"""
plt.figure(figsize=(14, 7))
colors = {'momentum_MSCI UK_6M': 'blue', 'momentum_MSCI US_6M': 'orange', 'momentum_TOPIX_6M': 'green'}
for key, color in colors.items():
    plt.plot(data_cleaned.index, data_cleaned[key], label=key, color=color)

    # add buying signal
    buy_signals = data_cleaned[data_cleaned['best_momentum_6M'] == key]
    plt.scatter(buy_signals.index, buy_signals[key], color=color, marker='^', s=80, label=f'Buy Signal for {key}', edgecolors='black')

plt.title('Three Indices Momentum and Buy Signals')
plt.xlabel('Year',fontsize=15)
plt.ylabel('6-Month Momentum',fontsize=15)
plt.legend(fontsize=11)
plt.grid(True)
plt.show()
# %%
