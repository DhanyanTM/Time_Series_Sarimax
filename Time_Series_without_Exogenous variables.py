#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import seaborn as sns
import math
import os
import pandas_gbq
pd.pandas.set_option('display.max_columns', None)
from random import random
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
import warnings







### Code to find P,D,Q value dynamically #########

def find_orders(ts):

    stepwise_model = auto_arima(ts['QTY'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=12,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True) # this works 

    return stepwise_model.order, stepwise_model.seasonal_order

## suppressing warnings #####
warnings.filterwarnings('ignore')

no_run_SKU=[]
pred=pd.DataFrame()


#######  Creating Date dataframe starting from the start period and ends with our requirment ##########

date_df=pd.DataFrame(pd.date_range('2018-01-01','2021-06-30', 
              freq='MS').strftime("%Y-%m-%d"))
date_df.columns={'date'}
date_df['date']=pd.to_datetime(date_df['date'])
date_df['Month'] = pd.to_datetime(date_df['date']).apply(lambda x: x.strftime('%m'))
date_df['YEAR'] = pd.to_datetime(date_df['date']).apply(lambda x: x.strftime('%Y'))


for brand in tx_data2.brand_name.unique():
    whole_data_brand=tx_data2[tx_data2['brand_name']==brand]
    
    ## ***************************************************************************
    ## Step 3.2: Channel Loop
    ## ***************************************************************************
    for channel in whole_data_brand.CHANNEL.unique():
        whole_data_chann=whole_data_brand[whole_data_brand['CHANNEL']==channel]

        ## ***************************************************************************
        ## Step 3.3: Region Loop
        ## ***************************************************************************
        for region in whole_data_chann.REGION_CODE.unique():

            whole_data_reg= whole_data_chann[whole_data_chann['REGION_CODE']==region]

            ## ***************************************************************************
            ## Step 3.4: Country Loop
            ## ***************************************************************************
            for con_code in whole_data_reg.COUNTRY_CODE.unique():

                whole_data_con= whole_data_reg[whole_data_reg['COUNTRY_CODE']==con_code]

                ## ***************************************************************************
                ## Step 3.5: watch_category Loop
                ## ***************************************************************************
                for watch_c in whole_data_con.watch_cat.unique():

                    whole_data_watch_cat= whole_data_con[whole_data_con['watch_cat']==watch_c]
                    
                ## ***************************************************************************
                ## Step 3.6: SKU Loop
                ## ***************************************************************************
    
                    for sku in whole_data_watch_cat.SKU.unique():
                        print('iteration for sku',sku)

                        SKU_df=whole_data_watch_cat[whole_data_watch_cat['SKU']==sku]
                        SKU_df['COUNTRY_CODE']==con_code
                        SKU_df['REGION_CODE']==region
                        SKU_df['watch_cat']==watch_c
                        SKU_df['CHANNEL']==channel
                        SKU_df['brand_name']==brand
                        
                    
                        print("SKU df******************* \n  ", SKU_df.head())
                    #  date_df2=date_df[date_df['date']<='2020-11-01']
                        whole_SKU_df=pd.merge(date_df,SKU_df,how='left',on=['date','Month','YEAR'])
                        test_df=whole_SKU_df[whole_SKU_df['date']<='2020-12-31']
                        test_df=test_df.fillna(0)
                        x=test_df[['QTY','date']]
                        
                        x=x.set_index('date')

                        x.asfreq('MS').index
                        
                        try:


                            order, seasonal_order = find_orders(x)
                            print('order',order)
                            print('seasonal_order',seasonal_order)
                            sarima =sm.tsa.statespace.SARIMAX(x['QTY'],order=order,freq='MS',seasonal_order=seasonal_order,
                                                         initialization='approximate_diffuse')
                            print('after model declaration')

                            #We can use SARIMAX model as ARIMAX when seasonal_order is (0,0,0,0) .
                            results = sarima.fit()
                            start=len(x)
                            end=len(x)+6
                            print('after length')
                                               
                            prediction = results.predict(start=start,end=end,typ='levels').rename('Forecast_QTY')
                            print('prediction done')
                            temp=pd.DataFrame(prediction)
                            temp['SKU']=sku
                            temp['Channel']=channel
                            temp['region']=region
                            temp['country_code']=con_code
                            temp['watch_cat']=watch_c
                            temp['brand_name']=brand
                            
                            
                            temp=temp.reset_index()
                    #         temp['REGION_CODE']=SKU_df.REGION_CODE.unique()[0]
                    #         temp['CHANNEL']=SKU_df.CHANNEL.unique()[0]
                            print(temp)


                            pred = pred.append(temp, ignore_index = True)
                            

                        except:

                            print(f'error for SKU---------------  {sku}')
                            no_run_SKU.append(sku)
                            
                            
                            
pred['Forecast_date']=pred['index']
pred=pred.drop(['index'],axis=1)
pred['Forecast_QTY']=pred['Forecast_QTY'].apply(np.ceil).astype(int)
pred.loc[pred['Forecast_QTY']<0, 'Forecast_QTY']=0


# In[5]:


pred


# In[3]:


pred.SKU.nunique()


# In[4]:


pred

