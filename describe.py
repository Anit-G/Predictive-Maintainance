import pandas as pd

def describe(data):
    des = data.describe()
    des.loc['median'] = data.median().values
    des.loc['coeffvariation'] = (data.std()/data.mean()).values
    des.loc['nunique'] = data.nunique().values
    des.loc['NullCount'] = data.isna().sum().values
    
    return des