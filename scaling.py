from sklearn.preprocessing import RobustScaler,MaxAbsScaler,MinMaxScaler,StandardScaler,Normalizer,QuantileTransformer
import pandas as pd
import numpy as np
def scaling(df,cols,scalers,single=True):
    
    scales = {'Robust_Scaler':RobustScaler,
               'MinMax': MinMaxScaler,
               'std':StandardScaler,
               'MaxAbs': MaxAbsScaler,
               'nor':Normalizer,
               'QT':QuantileTransformer}
    if single:
        transformer = scales[scalers]
        df_transform =  pd.DataFrame(transformer().fit_transform(df[cols]), 
                                    columns=cols, 
                                    index= df.index)
        print(f'single: {single}')
        print(f"Columns: {cols}")
        print(f"Transformer: {transformer}")
        print()
    else:
        data = []
        for col,scaler in zip(cols,scalers):
            transformer = scales[scaler]
            df_t =  pd.DataFrame(transformer().fit_transform(df[col]), 
                                    columns=col, 
                                    index= df.index)
            data.append(df_t)
            print(f"Columns: {col}")
            print(f"Transformer: {transformer}")
            #print(f"Range of values: {df_t.max()}")
            print()

        df_transform = pd.concat(data,axis=1)

    return df_transform
