import streamlit as st
import pandas as pd
import datetime
import numpy as np
from os.path import dirname, join 
import sys 
sys.path.insert(0, join(dirname(__file__), '..')) 

#from Ratemaking import load_data,_df,months_between,AdjustedLosses,AdjustedPrem

st.set_page_config(page_title="Trending", page_icon="🎢", layout="wide")

"""# PAGE IN DEVELOPMENT"""

# """# Trending Loss Ratios

# ### We are getting data related to Annual Inflation Rates by country from World Bank's website: [data.worldbank.org](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=US&view=chart)
# """

# # Lets work on Inflation Rates first
# filepath = "ratemaking_with_python/605_InflationRates.csv"

# # load the inflation dataset
# inflation_rates = load_data(filepath)
# inflation_rates

# # inflation rates in USA
# inf_us = inflation_rates[inflation_rates['Country Name'] == "United States"]
# st.write("We extract the inflation rates for the USA:",inf_us)

# inf_index = {}
# start = 1988; end = 1997
# for i in range(start, end+1):
#     inf_index[i] = inf_us.iloc[0, i-1960+4]

# st.write("Inflation indeces from 1988 to 1997 in the US are:\n")
# _df(inf_index, "Inflation rate")

# # average inflation rates
# inf_avg = {}
# periods = []
# keys = list(inf_index.keys())
# for i in range(0,len(keys)):
#     avg=0
#     temp = str(keys[i])+"-1997"
#     for j in range(i,len(keys)):
#         avg+= inf_index[keys[j]]
#     periods.append(temp)
#     inf_avg[keys[i]] = avg/(j-i+1)

# st.write("The average inflation rates are:")
# inf_avg_df = pd.DataFrame({
#     "periods":periods,
#     "average inflation":inf_avg.values()
# })
# st.dataframe(inf_avg_df,hide_index=True)


# """## Our Assumptions are:
# ### --> Policies are written uniformly over time.
# ### --> Premiums are earned uniformly over the policy period.
# ### --> Losses occur uniformly over the policy period.
# ### --> Policies have annual terms.

# ## Trend losses for inflation.
# ##### Our experience periods are the historical accident years from 1988 to 1997.
# ##### We assume future policy period begins on Jan 1, 1998 and inflation rate will be in effect for 12 months. Thus our forecast period average accident date is:
# ##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
# """
# col20, col21, col22 = st.columns(3)
# loss_inf_period = {}
# loss_forecast_Date = datetime.date(1999,1,1)
# for i in inf_index.keys():
#     expDate = datetime.date(i,7,1)
#     diff = months_between(loss_forecast_Date,expDate)
#     loss_inf_period[i] = diff
# col20.write("The trending periods for losses are:")
# loss_inf_period_df = pd.DataFrame({
#     "periods":periods,
#     "trending periods":loss_inf_period.values()
# })
# col20.dataframe(loss_inf_period_df,hide_index=True)

# # trend factors for losses
# loss_inf_factor = {}
# for i in loss_inf_period.keys():
#     loss_inf_factor[i] = (1 + (0.01*inf_avg[i]))**loss_inf_period[i]
# col21.write("The trend factors for losses are:")
# loss_inf_factor_df = pd.DataFrame({
#     "periods":periods,
#     "trend factors":loss_inf_factor.values()
# })
# col21.dataframe(loss_inf_factor_df,hide_index=True)

# # Now we trend the losses
# inf_trendedLosses = {}
# for i in loss_inf_factor.keys():
#     inf_trendedLosses[i] = AdjustedLosses[i]*loss_inf_factor[i]
# st.write("Losses trended for inflation:")
# _df(inf_trendedLosses,"Inflation Trended Losses")



# """## Trend Premiums for inflation.
# ##### Trend will be estimated from earned premium data. The trend period will be from the average earned date in each historical period to the average earned date at the new rate level. Because of the uniform assumption, the average earned date of a period is the midpoint of the first and last dates that premiums could be earned in that period. So, these dates will depend on the policy term length.
# ##### Future policy period begins in Jan 1, 1998. Inflation rate will be in effect for 12 months. Thus our forecast period average earned date is:
# ##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
# """
# col23, col24, col25 = st.columns(3)
# prem_inf_period = {}
# prem_forecast_Date = datetime.date(1999,1,1)
# for i in inf_index.keys():
#     expDate = datetime.date(i,1,1)
#     diff = months_between(prem_forecast_Date,expDate)
#     prem_inf_period[i] = diff
# col23.write("The trending periods for premiums are:")
# prem_inf_period_df = pd.DataFrame({
#     "periods":periods,
#     "trending periods":prem_inf_period.values()
# })
# col23.dataframe(prem_inf_period_df,hide_index=True)

# # trend factors for premiums
# prem_inf_factor = {}
# for i in prem_inf_period.keys():
#     prem_inf_factor[i] = (1 + (0.01*inf_avg[i]))**prem_inf_period[i]
# col24.write("The trend factors for premiums are:")
# prem_inf_factor_df = pd.DataFrame({
#     "periods":periods,
#     "trend factors":prem_inf_factor.values()
# })
# col24.dataframe(prem_inf_factor_df,hide_index=True)

# # Now we trend the premiums
# inf_trendedPrems = {}
# for i in prem_inf_factor.keys():
#     inf_trendedPrems[i] = AdjustedPrem[i]*prem_inf_factor[i]
# st.write("Premiums trended for inflation:")
# _df(inf_trendedPrems,"Inflation Trended Premiums")