import pandas as pd
import numpy as np
import scipy.stats 
from scipy.stats import norm


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Returns a DataFrame with colums:
    for the wealth index, the previos peaks, and
    the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/wealth_index
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previos Peak": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles 
    by MarketCap
    """
    me_m = pd.read_csv("/Users/iuliia/Desktop/data/Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Load and Format the EHEC Hedge Fund Index
    """
    hfi = pd.read_csv("/Users/iuliia/Desktop/data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series of a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not. 
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    .jarque_bera returns a tuple
    """ 
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -(np.percentile(r, level))
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        

        
def var_gaussian(r, level=5):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the Z score assuming it was Gaussian
    
    # Returns the Z-score
    z = norm.ppf(level/100) # at 5% level.
    
    # Z-score tells us how far it is from the mean.
    return -(r.mean() + z*r.std(ddof=0))



def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the Z score assuming it was Gaussian
    
    # Returns the Z-score
    z = norm.ppf(level/100) # at 5% level.
    if modified:
        # modify the Z-score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 -1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    
    # Z-score tells us how far it is from the mean.
    return -(r.mean() + z*r.std(ddof=0))



def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
