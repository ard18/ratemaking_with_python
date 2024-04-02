import streamlit as st
import pandas as pd
import datetime
import numpy as np
from os.path import dirname, join 
import sys 
sys.path.insert(0, join(dirname(__file__), '..')) 

from Ratemaking import loss_data,_df,ULT_LOSSES

st.set_page_config(page_title="Adjusting", page_icon="🎢", layout="wide")

"""# PAGE IN DEVELOPMENT"""

# """# Calculating Rate and Benefit Adjustment Factors
# We use a simple general formula derived by Richard A. Bill for automating the calculation of rate and benefit adjustment factors. This is based on the parallelogram method. More details on the formula can be found in the paper by Richard A. Bill on:
# https://www.casact.org/abstract/generalized-earned-premium-rate-adjustment-factors

# We exclude any fluctuations arising due to legal changes.
# ## Adjusting Premiums for Rate Changes
# """

# # Net Premium Earned (Earned Premium - Ceded Earned Premium(or Reinsurance costs))
# net_prem_earned = {}
# for i in range(1988,1998):
#     net_prem_earned[i] = list( loss_data[loss_data['AccidentYear']==i]['EarnedPremNet_D'] )[0]
# st.subheader("Earned Premium (Net)")
# _df(net_prem_earned,"Earned Premium Net")

# col10,col11,col12 = st.columns(3)

# col10.subheader("Assume some rate changes (Other rate changes can be assumed).")

# # Assume rate changes (for now, values are taken similar to those in Massachusetts Rate Filings)
# rate_changes = {
#             datetime.date(1988,4,1):0.198, #datetime.date(1989,1,1):0.1,
#                 datetime.date(1990,7,1):0.262, #datetime.date(1991,4,1):-0.04,
#             datetime.date(1991,5,1):0.113,  #datetime.date(1992,3,1):0.07,
#             datetime.date(1993,8,1):0.062, #datetime.date(1994,2,1):0.08,
#             datetime.date(1996,5,1):-0.122
#                 }
# rate_changes_df = pd.DataFrame({
#     "rate change dates":rate_changes.keys(),
#     "rate changes ":rate_changes.values()
# })
# col10.dataframe(rate_changes_df, hide_index=True)
# # first calculate the rate change indeces
# rates = list(rate_changes.values())
# rate_index =[1.00]+[ (1+i) for i in rates ] # including initial index of segment without changes = 1.00 (rate change = 0%)
# rate_index_df = pd.DataFrame({
#     'rate index':rate_index
# })
# col11.dataframe(rate_index_df,hide_index=True)

# cum_index = []
# f = 1
# for i in rate_index:
#     f *= i
#     cum_index.append( round(f, 4))
# cum_index_df = pd.DataFrame({
#     'cumulative rate index':cum_index
# })
# col12.dataframe(cum_index_df,hide_index=True)

# current_cum_rate_index = cum_index[-1]
# st.write("Current Cumulative Rate Level Index =",current_cum_rate_index)

# # To calculate the portions earned by premiums under each rate change
# T = 1; E = 1;

# def months_between(date1,date2):
#     '''This function calculates the difference between 2 given dates in months
#     date1, date2 are in datetime.date() format'''
#     m1=date1.year*12+date1.month
#     m2=date2.year*12+date2.month
#     months=m1-m2    # difference between the dates

#     return months/12



# def find_remains(rate_dates, earned_prem_year, L):
#     '''This function calculates the remaining portions of earned premium under the rate changes
#        rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
#     if L!=[]:
#         L.append(0) # appending 0 as a means for calculating the last portion
#         to_return = []  # the list that contains the portions
#         max = 1 # maximum value (total area of an year of earned premium)
#         for i in range(0, len(L)):
#             if L[i]!=0:
#                 diff = max - L[i]   # calculate remaining portion
#                 to_return.append(round( diff,5))
#                 max = L[i]
#                 if L[i+1]==0:   # for the last portion to be appended
#                     to_return.append(round( max,5))
#             else:
#                 to_return.append(0)

#         if to_return.count(0) == len(to_return):
#             to_return = earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, to_return)

#         to_return.pop()
#         return to_return



# def earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, L):
#     '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
#     rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
#     c = 0
#     start_date = datetime.date(earned_prem_year,1,1)
#     for i in rate_dates:
#         if( months_between(i, start_date)>0 ):      # checking where to insert 1
#             break
#         else:
#             c+=1
#     L.insert(c,1)   # insert 1 as portion earned by premium
#     return L



# def earnedPortion(rate_dates, earned_prem_years):
#     '''This function calculates the portion of earned premium under given rate changes
#     rate_dates is a dictionary containing the dates of rate changes and earned_prem_years is also a dictionary containing the years in which premium is earned'''
#     portion = {}
#     for i in earned_prem_years:
#         portion[i] = []
#     for i in earned_prem_years:
#         start_date = datetime.date(i,1,1)
#         for j in rate_dates:

#             if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
#                 # algorithm for calculating portions of earned premium
#                 D = months_between(j, start_date)

#                 A = D+T
#                 B = max( A-E, 0 )
#                 C = max( D, 0 )

#                 P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
#                 portion[i].append( round(P, 5))
#             else:
#                 portion[i].append(0)

#     for i in portion.keys():
#         portion[i] = find_remains(rate_dates, i, portion[i])
#     return(portion)


# rate_effec_dates = list( rate_changes.keys())
# years_toAdjust = list( net_prem_earned.keys() )
# earned_NetPremPortion = earnedPortion(rate_effec_dates, years_toAdjust)
# earned_NetPremPortion_df = pd.DataFrame({
#     "Accident Years":earned_NetPremPortion.keys(),
#     "Earned Premium (Net) Portions": earned_NetPremPortion.values()
# })
# st.subheader("The portion earned by the premium in the years w.r.t. the rate changes are:\n")
# st.dataframe(earned_NetPremPortion_df,hide_index=True)

# # Average Cumulative Rate Level Indices
# def AvgCumulIndices(L, cumul_indices):
#     '''This function calculates the average cumulative rate level indices for the earned premium
#     L, cumul_indices are numpy arrays where L contains the portions of earned premiums and cumul_indices contains the cumulative rate level indices'''
#     prod = L*cumul_indices
#     sum = 0
#     for i in prod:
#         sum+=i
#     return round(sum, 5)

# # changing to numpy arrays
# for i in earned_NetPremPortion.keys():
#     earned_NetPremPortion[i] = np.array(earned_NetPremPortion[i])
# cum_index = np.array(cum_index)

# avg_CumulIndices = {}
# for i in earned_NetPremPortion.keys():
#     avg_CumulIndices[i] = AvgCumulIndices(earned_NetPremPortion[i], cum_index)
# avg_CumulIndices_df = pd.DataFrame({
#     "Accident Year": avg_CumulIndices.keys(),
#     "Average Cumulative Rate Level Indices":avg_CumulIndices.values() 
# })
# st.subheader("Average Cumulative Rate level Indices for the respective accident years")
# st.dataframe(avg_CumulIndices_df, hide_index=True)

# col13, col14 = st.columns(2)
# # On-Level Factors for the premiums
# onlevel = {}
# for i in avg_CumulIndices.keys():
#     onlevel[i] = round( current_cum_rate_index/avg_CumulIndices[i], 5 )
# onlevel_df = pd.DataFrame({
#     "Accident Year": onlevel.keys(),
#     "On-Level Factors":onlevel.values() 
# })
# col13.subheader("On-Level Factors for the premiums")
# col13.dataframe(onlevel_df, hide_index=True)

# # On-Levelling the Premiums
# AdjustedPrem = {}
# for i in onlevel.keys():
#     AdjustedPrem[i] = round( net_prem_earned[i] * onlevel[i], 5)
# AdjustedPrem_df = pd.DataFrame({
#     "Accident Year": AdjustedPrem.keys(),
#     "Rate Level Adjusted Premiums(Net)":AdjustedPrem.values() 
# })
# col14.subheader("Premiums adjusted for rate level changes:")
# col14.dataframe(AdjustedPrem_df, hide_index=True)







# """## Adjusting Losses for Benefit Changes"""
# col15,col16,col17 = st.columns(3)
# col15.subheader("""Assume some benefit changes (Other benefit changes can be assumed).""")
# # Assume benefit changes (Other benefit changes can be assumed)
# benefit_changes = {
#             # datetime.date(1988,4,1):0.05,
#                  datetime.date(1989,1,1):0.1,
#             # datetime.date(1990,7,1):-0.02,
#                  datetime.date(1991,4,1):-0.04,
#             # datetime.date(1991,5,1):0.11,
#                  datetime.date(1992,3,1):0.07,
#             # datetime.date(1993,8,1):-0.05,
#                  datetime.date(1994,2,1):0.08,
#             # datetime.date(1996,8,1):0.15
#                 }
# benefit_changes_df = pd.DataFrame({
#     "benefit change dates":benefit_changes.keys(),
#     "benefit changes ":benefit_changes.values()
# })
# col15.dataframe(benefit_changes_df, hide_index=True)
# # first calculate the benefit change indeces
# benefits = list(benefit_changes.values())
# benefit_index =[1.00]+[ (1+i) for i in benefits ] # including initial index without changes = 1.00 (rate change = 0%)
# benefit_index_df = pd.DataFrame({
#     "Benefit index":benefit_index
# })
# col16.dataframe(benefit_index_df, hide_index=True)

# loss_lvl = []
# f = 1
# for i in benefit_index:
#     f *= i
#     loss_lvl.append( round(f, 4))
# loss_lvl_df = pd.DataFrame({
#     'loss level index':loss_lvl
# })
# col17.dataframe(loss_lvl_df,hide_index=True)

# current_loss_lvl = loss_lvl[-1]
# st.write("Current Loss Level Index =",current_loss_lvl)

# T = 1; E = 1;

# def months_between(date1,date2):
#     '''This function calculates the difference between 2 given dates in months
#     date1, date2 are in datetime.date() format'''
#     m1=date1.year*12+date1.month
#     m2=date2.year*12+date2.month
#     months=m1-m2    # difference between the dates

#     return months/12



# def find_remains(ben_dates, loss_year, L):
#     '''This function calculates the remaining portions of earned premium under the rate changes
#        ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
#     if L!=[]:
#         L.append(0) # appending 0 as a means for calculating the last portion
#         to_return = []  # the list that contains the portions
#         max = 1 # maximum value (total area of an year of losses)
#         for i in range(0, len(L)):
#             if L[i]!=0:
#                 diff = max - L[i]   # calculate remaining portion
#                 to_return.append(round( diff,5))
#                 max = L[i]
#                 if L[i+1]==0:   # for the last portion to be appended
#                     to_return.append(round( max,5))
#             else:
#                 to_return.append(0)

#         if to_return.count(0) == len(to_return):
#             to_return = Portion_ForUnaffectedYear(ben_dates, loss_year, to_return)

#         to_return.pop()
#         return to_return



# def Portion_ForUnaffectedYear(ben_dates, loss_year, L):
#     '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
#     ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
#     c = 0
#     start_date = datetime.date(loss_year,1,1)
#     for i in ben_dates:
#         if( months_between(i, start_date)>0 ):      # checking where to insert 1
#             break
#         else:
#             c+=1
#     L.insert(c,1)   # insert 1 as portion earned by premium
#     return L



# def LossPortion(ben_dates, loss_years):
#     '''This function calculates the portion of earned premium under given rate changes
#     ben_dates is a list containing the dates of benefit changes and loss_years is also a list containing the years in which losses occur'''
#     portion = {}
#     for i in loss_years:
#         portion[i] = []
#     for i in loss_years:
#         start_date = datetime.date(i,1,1)
#         for j in ben_dates:

#             if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
#                 # algorithm for calculating portions of earned premium
#                 D = months_between(j, start_date)

#                 A = D+T
#                 B = max( A-E, 0 )
#                 C = max( D, 0 )

#                 P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
#                 portion[i].append( round(P, 5))
#             else:
#                 portion[i].append(0)

#     for i in portion.keys():
#         portion[i] = find_remains(ben_dates, i, portion[i])
#     return(portion)


# ben_effec_dates = list( benefit_changes.keys())
# years_toAdjust = list( ULT_LOSSES.keys() )
# LossesPortion = LossPortion(ben_effec_dates, years_toAdjust)
# LossesPortion_df = pd.DataFrame({
#     "Accident Years":LossesPortion.keys(),
#     "Loss Portions": LossesPortion.values()
# })
# st.subheader("The portion of the losses in the years w.r.t. the benefit changes are:\n")
# st.dataframe(LossesPortion_df,hide_index=True)


# # Average Loss Levels
# def AvgLossLevel(L, loss_levels):
#     '''This function calculates the average Loss levels for the historical periods
#     L, loss_levels are numpy arrays where L contains the portions of losses and loss_levels contains the loss levels'''
#     prod = L*loss_levels
#     sum = 0
#     for i in prod:
#         sum+=i
#     return round(sum, 5)

# for i in LossesPortion.keys():
#     LossesPortion[i] = np.array(LossesPortion[i])
# loss_lvl = np.array(loss_lvl)

# avg_LossLvl = {}
# for i in LossesPortion.keys():
#     avg_LossLvl[i] = AvgLossLevel(LossesPortion[i], loss_lvl)
# avg_LossLvl_df = pd.DataFrame({
#     "Accident Year": avg_LossLvl.keys(),
#     "Average Loss Level Indices":avg_LossLvl.values() 
# })
# st.subheader("Average Loss level Indices for the respective accident years")
# st.dataframe(avg_LossLvl_df, hide_index=True)

# col18, col19 = st.columns(2)
# # Adjustment Factors
# adjusts = {}
# st.write("Current Loss Level =",current_loss_lvl)
# for i in avg_LossLvl.keys():
#     adjusts[i] = round( current_loss_lvl/avg_LossLvl[i], 5 )
# adjusts_df = pd.DataFrame({
#     "Accident Year": adjusts.keys(),
#     "Adjustment Factors":adjusts.values() 
# })
# col18.subheader("Adjustment Factors for the losses")
# col18.dataframe(adjusts_df, hide_index=True)

# # Adjusting the Losses
# AdjustedLosses = {}
# for i in adjusts.keys():
#     AdjustedLosses[i] = round( ULT_LOSSES[i] * adjusts[i], 5)
# AdjustedLosses_df = pd.DataFrame({
#     "Accident Year": AdjustedLosses.keys(),
#     "Benefit Adjusted Losses ":AdjustedLosses.values() 
# })
# col19.subheader("Losses adjusted for benefit changes:")
# col19.dataframe(AdjustedLosses_df, hide_index=True)