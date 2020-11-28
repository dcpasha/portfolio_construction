# We use this extension to reload modules automatically.
# Magic command starts with a percentage 
%load_ext autoreload
%autoreload 2

# To import any modules including that I created
import edhec_risk_kit as erk
import pandas as pd

# Concat => concatenates things together in columns because I specified axis = 'columns'  
pd.concat([hfi.mean(), hfi.median(),hfi.mean()>hfi.median()], axis='columns')
# If mean < median, then we have a negative skew.

# Generates the Excessive curtosis = curtosis - kurtosis
scipy.stats.kurtosis(normal_rets)


# It applies it to the entire dataframe at once.
erk.is_normal(hfi)
# But if we want to apply f-n(is_normal) to every column => .aggregate 
hfi.aggregate(erk.is_normal)
