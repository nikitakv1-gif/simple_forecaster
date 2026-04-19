freq_to_seasonal_periods = {
    # yearly
    "Y": 1,
    "A": 1,
    # quarterly
    "Q": 4,
    "QS": 4,
    # monthly
    "M": 12,
    "MS": 12,
    "ME": 12,
    # weekly
    "W": 52,
    # daily
    "D": 7,  
    "B": 5, 
    # hourly
    "H": 24,
    # minutely
    "T": 60,  
    "min": 60,
    # secondly
    "S": 60,
}
def seasonality(freq):
    return freq_to_seasonal_periods[freq]
