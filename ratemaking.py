
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2
import datetime
import statsmodels.api as sm
import warnings
pd.set_option("display.max_columns",None)

st.set_page_config(
    page_title="Ratemaking",
    page_icon="✌️",
    layout="wide"
)
# display various dataframes
def _df(ds, val_col):
    ds_df = pd.DataFrame({
        "Accident Year":ds.keys(),
        val_col: ds.values()
    })
    st.dataframe(ds_df, hide_index=True)

# home page configs 
st.title("Worker's Compensation 🏢👨🏻‍💼")
st.subheader("Pricing(Ratemaking) Worker's Compensation Premiums using Actuarial Techniques")

# our csv file
filepath = "./wkcomp_pos.csv"

# load the dataset
@st.cache_data # for faster execution
def load_data(filepath):
    df = pd.DataFrame(pd.read_csv(filepath))
    return df
'''##### - This is our dataset'''
dataset = load_data(filepath)
st.dataframe(dataset) # the worker's compensation dataset

# dataset columns
columns = dataset.columns
st.write("Features in dataset:",columns)

# correlation heatmap
st.subheader("Correlation heatmap")
df_corr = dataset.drop(columns=['GRCODE','GRNAME'])
fig, ax = plt.subplots(figure=(15,15))
sns.heatmap(df_corr.corr(), ax=ax, annot=True, linewidths=0.36, linecolor="black", fmt=".2f")
st.write(fig)

"""We see that there's a strong positive correlation between the following features:
- PostedReserve97_D with IncurLoss_D, CumPaidLoss_D, EarnedPremDIR_D and EarnedPremNet_D  
- EarnedPremNet_D with CumPaidLoss_D and IncurLoss_D
"""

# display companies in the dataset
col1, col2 = st.columns(2)
dataset["GRCODE-GRNAME"] = dataset['GRCODE'].astype('str')+"-"+dataset['GRNAME']
companies = dataset["GRCODE-GRNAME"]
companies = pd.DataFrame({'Companies':pd.unique(companies)})
with col1:
    st.subheader("Companies in the dataset:")
    st.dataframe(companies, hide_index=True, width = 300)

# sample of 5 companies chosen for this project
sample_companies = ["Allstate Ins Co Grp", "California Cas Grp", "Celina Mut Grp", "Federal Ins Co Grp", "Farm Bureau of MI Grp"]
grcodes = [86, 337, 353, 388, 671]

df_comp = pd.DataFrame(
    {
        'GRCODE':grcodes,
        'NAME':sample_companies,
    }
)
with col2:
    st.subheader("For this project, we use a sample of 5 companies:")
    st.dataframe(df_comp, hide_index=True,)

# select a grcode
slt_comp = st.selectbox("Select a company by GRCODE:", grcodes, index=0, placeholder="Choose an option")


"""# Let's see some Triangles """

# python class that consists of 4 different averaging methods for averaging loss-development factors
class AveragingMethods:
    def __init__(self, data):
        '''Here, Data is of type list'''
        self.data = data
    def SimpleAvg(self): # simple average
        sum = 0
        for i in self.data:
            sum += i
        return round( sum/len(self.data), 4)
    def VolumeAvg(self, dt1, dt2): # volume-weighted average
        sum1 = 0
        for i in dt1:
            sum1 += i
        sum2 = 0
        for j in dt2:
            sum2 += j
        return round( sum1/sum2,4)
    def MedialAvg(self): # medial average
        minimum = min(self.data)
        maximum = max(self.data)
        sum = 0
        if len(self.data) > 2:
            for i in self.data:
                sum += i
            sum -= (maximum+minimum)
            return round( sum/(len(self.data)-2),4)
        else:
            return  round( (maximum+minimum)/2,4)
    def GeometricAvg(self): # geometric average
        sum = 1
        for i in self.data:
            sum *= i
        return round( sum**(1/len(self.data)),4)


def LossData(grcode):
    '''This function extracts the loss data of a specific company corresponding to its GRCODE
        Here data is of type: dataframe'''
    company = dataset[dataset["GRCODE"]==grcode]
    return(company)


def createLossTriangle(data):
    '''This function extracts and creates Loss triangles
        Here data is of type: dataframe'''
    trframe = {}      # dict containing loss triangle values for various accident years
    for i in range(1988,1998):
        L = []
        for j in range(i,1998):
            condition = ( (data['AccidentYear']==i) & (data['DevelopmentYear']==j) )
            L.append(int(data.loc[condition]['CumPaidLoss_D']))
        i = int(i)
        trframe[i] = L
    return trframe


def displayTriangleData(data):
    '''This function displays Loss Triangle data
       Here data is of type: dictionary'''
    for i in data.keys():
        print(i, end = "\t\t")
        for j in data[i]:
            print(j, end = "\t")
        print("\n")


def computeLDF(data):
    '''This function computes Loss Development Factors
       Here data is of type: dictionary'''
    trframe = {}
    for i in data.keys():
        L = []
        for j in range(len(data[i])-1):
            ldf = data[i][j+1]/data[i][j]
            L.append( round(ldf,4) )
        i = int(i)
        trframe[i] = L
    return trframe


def computeAverageLDF(ldf_info, loss_info):
    '''This function computes various Averages of Loss Development Factors
       Here data is of type: dictionary'''
    DK = list(ldf_info.keys())
    DK = sorted(DK, reverse=True)
    trframe = {
        'SimpleAvg':[],
        'VolumeAvg':[],
        'MedialAvg':[],
        'GeometricAvg':[]
    }
    # for Medial, Simple and Geometric Averages
    for i in range(0,10):
        L = []
        c = 1
        for j in DK:
            try:    # to avoid Index Out of Bounds
                if ldf_info[j][i] and c<=5:
                    L.append(ldf_info[j][i])
                    c+=1
            except:
                pass
        if(L!=[]):
            obj = AveragingMethods(L)               # object of class Averaging methods
            simp_avg = obj.SimpleAvg()
            med_avg  = obj.MedialAvg()
            geo_avg  = obj.GeometricAvg()
            trframe['SimpleAvg'].append(simp_avg)
            trframe['MedialAvg'].append(med_avg)
            trframe['GeometricAvg'].append(geo_avg)
    # only for Volume-Weighted Average
    for i in range(1,10):
        L1 = []
        L2 = []
        c = 1
        for j in DK:
            try:
                if loss_info[j][i] and loss_info[j][i-1] and c<=5:
                    L1.append(loss_info[j][i])
                    L2.append(loss_info[j][i-1])
                    c+=1
            except:
                pass
        if(L1!=[] and L2!=[]):
            obj = AveragingMethods(L1)
            vol_avg = obj.VolumeAvg(L1, L2)
            trframe['VolumeAvg'].append(vol_avg)
    return trframe

# the loss data of selected company
loss_data = LossData(slt_comp)
st.write("Loss data of selected company:",loss_data)

# the loss development triangle
st.write("Loss Development Triangle")
loss_triangle = createLossTriangle(loss_data)
# Determine the maximum length of the arrays
max_length = max(len(arr) for arr in loss_triangle.values())
# Pad the arrays with NaN to make them all the same length
data_padded = {key: arr + [np.nan] * (max_length - len(arr)) for key, arr in loss_triangle.items()}
# Create DataFrame from the padded dictionary
loss_df = pd.DataFrame.from_dict(data_padded)
loss_df = loss_df.T
for i in range(0,max_length):
    loss_df.rename(columns={i:(i+1)*12,}, inplace=True)
st.dataframe(loss_df,width=1000)


# triangle of LDFs
st.write("Loss Development Factors")
ldf_triangle = computeLDF(loss_triangle)
# Determine the maximum length of the arrays
max_length = max(len(arr) for arr in ldf_triangle.values())
# Pad the arrays with NaN to make them all the same length
data_padded = {key: arr + [np.nan] * (max_length - len(arr)) for key, arr in ldf_triangle.items()}
# Create DataFrame from the padded dictionary
df = pd.DataFrame.from_dict(data_padded)
df = df.T
for i in range(0,max_length):
    df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
st.dataframe(df,width=1000)

# averages of ldf
avg_ldf = computeAverageLDF(ldf_triangle, loss_triangle)
col3,col4 = st.columns(2, gap="large")

# Plot line chart
fig = go.Figure()
# Add lines for each list
for key, values in avg_ldf.items():
    fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode='lines', name=key))
# Customize layout
fig.update_layout(title='Trend in Averages',
                  xaxis_title='Data Points', yaxis_title='Values')
# Display the chart
col4.plotly_chart(fig)

avg_ldf_df = pd.DataFrame(
    {
    'Simple Average':avg_ldf['SimpleAvg'],
    'Medial Average':avg_ldf['MedialAvg'],
    'Volume-Weighted':avg_ldf['VolumeAvg'],
    'Geometric Average':avg_ldf['GeometricAvg'],
    })
avg_ldf_df = avg_ldf_df.T
for i in range(0,max_length):
    avg_ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
col3.subheader("Averages of LDFs")
col3.dataframe(avg_ldf_df)

# Select LDF.
ldf_choices = list(avg_ldf.keys())
chosen_Ldf = st.selectbox("Select an averaging method for the LDFs:", ldf_choices, index=0, placeholder="Choose an option")
selected_Ldf = avg_ldf[chosen_Ldf]

# We select an arbitrary tail factor
tail = 1.0000
selected_Ldf.append(tail)
selected_Ldf = selected_Ldf[::-1] # revert the list for finding CDFs

selected_Ldf_df = pd.DataFrame({chosen_Ldf:selected_Ldf[::-1],})
selected_Ldf_df = selected_Ldf_df.T
for i in range(0,max_length+1):
    if i==max_length:
        selected_Ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,'ult'),}, inplace=True)
    else:
        selected_Ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
st.subheader("Selected LDFs")
st.dataframe(selected_Ldf_df)

# Cumulative Loss Development factors
cdf = []
for i in range(1, len(selected_Ldf)+1):
    f = 1
    for j in range(0, i):
        f*=selected_Ldf[j]
    cdf.append( round( f,4) )
cdf_df = pd.DataFrame({'CDF':cdf[::-1],})
cdf_df = cdf_df.T
for i in range(0,max_length+1):
        cdf_df.rename(columns={i:"{}-{}".format((i+1)*12,'ult'),}, inplace=True)
st.subheader("Cumulative Development Factors")
st.dataframe(cdf_df)

# Projected Ultimate Losses
proj_ultLosses = {}
for i in range(0, len(cdf)):
    for j in range(0, len(loss_triangle)):
        if(i==j):
            proj_ultLosses[ list(loss_triangle.keys())[j] ] = round( list(loss_triangle.values())[i][-1]*cdf[i],4)
st.subheader("Chain-Ladder Projected Ultimate Losses")
_df(proj_ultLosses,"Chain-Ladder Projected Ultimate Losses")

'''## Using GLMs for projecting Ultimate Losses'''
def GLM_UltClaims(dataset):
    '''The dataset used here is a dataframe.
    This function outputs a dataframe having projected the lower half of the original dataset, i.e., projected losses'''
    # dataset is a dataframe
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    i = 0
    k = 9
    while i<9:
        x_train = [ dataset.iloc[j,i] for j in range(0,k)   ]
        y_train = [ dataset.iloc[j,i+1] for j in range(0,k) ]
        x_test =  [ dataset.iat[j,i] for j in range(k,10)   ]

        glm_model = sm.GLM(y_train,x_train,family=sm.families.Gaussian()) # using gaussian glm
        model_results = glm_model.fit(method="bfgs")
        y_test = model_results.predict(x_test)

        for j in range(0,len(y_test)):
            dataset.iat[k+j,i+1] = y_test[j]
        i+=1
        k-=1
    return dataset
data = GLM_UltClaims(loss_df)
st.write("Dataframe with GLM projected losses in the lower triangle")
data
# GLM projected ultimate Losses
glmUlt_Losses = {}
for i in range(0,len(loss_triangle.keys())):
    glmUlt_Losses[list(loss_triangle.keys())[i]] = data.iat[i,9]
_df(glmUlt_Losses,"GLM Projected Ultimate Losses")

"""## Lets evaluate the closeness of our projected ultimate losses to the actual ultimate losses."""

# metrics used: mean absolute error, and r^2 coefficient

# Actual Ultimate Losses
act_ultLosses = {}
for i in range(1988,1998):
        condition = ( (loss_data['AccidentYear']==i) & (loss_data['DevelopmentLag']==10) )
        act_ultLosses[i] = int( loss_data.loc[condition]['CumPaidLoss_D'])

col5, col6, col7 = st.columns(3)
col5.subheader("Actual Ultimate Losses")
col5.dataframe(act_ultLosses, width=300)
col6.subheader("Chain-Ladder Projected Ultimate Losses")
col6.dataframe(proj_ultLosses, width=300)
col7.subheader("GLM Projected Ultimate Losses")
col7.dataframe(glmUlt_Losses, width=300)

mae1 = mae(list(act_ultLosses.values()), list(proj_ultLosses.values()))
mae2 = mae(list(act_ultLosses.values()), list(glmUlt_Losses.values()) )

col8, col9 = st.columns(2)
with col9:
    st.write( "\nMean Absolute Error based on GLM values =",round( mae2,3))
    st.write("\nR^2 coefficient based on GLM values =", round( r2(list(act_ultLosses.values()), list(glmUlt_Losses.values() ) ),3) )
with col8:
    st.write("Mean Absolute Error based on Chain-Ladder values =", round( mae1,3))
    st.write("R^2 coefficient based on Chain-Ladder values =", round( r2(list(act_ultLosses.values()), list(proj_ultLosses.values()) ),3))

#"""The R^2 coefficients are close to 1, which is very good. This means that both GLM and Chain-Ladder Method provide a good fit between the projected values and actual values."""
"""### Select the losses of the method producing lower MAE"""
losses_selected = ""
if mae1<mae2:
    st.subheader("Chain-Ladder")
    losses_selected = "Chain-Ladder Projected Ultimate Losses"
    ULT_LOSSES = proj_ultLosses
else:
    st.subheader("GLM")
    losses_selected = "GLM Projected Ultimate Losses"
    ULT_LOSSES = glmUlt_Losses
_df(ULT_LOSSES,losses_selected )

ultLosses_dict = ({
    "Chain-Ladder":proj_ultLosses.values(),
    "GLM":glmUlt_Losses.values(),
    "Actual":act_ultLosses.values()
})
# Plot line chart
fig2 = go.Figure()
# Add lines for each list
for key, values in ultLosses_dict.items():
    fig2.add_trace(go.Scatter(x=list(range(1988, 1988*(len(values) + 1))), y=list(values), mode='lines', name=key))
# Customize layout
fig2.update_layout(title='Ultimate Losses',
                  xaxis_title='Accident Years', yaxis_title='Losses')
# Display the chart
st.plotly_chart(fig2)






"""# Calculating Rate and Benefit Adjustment Factors
We use a simple general formula derived by Richard A. Bill for automating the calculation of rate and benefit adjustment factors. This is based on the parallelogram method. More details on the formula can be found in the paper by Richard A. Bill on:
https://www.casact.org/abstract/generalized-earned-premium-rate-adjustment-factors

We exclude any fluctuations arising due to legal changes.
## Adjusting Premiums for Rate Changes
"""

# Net Premium Earned (Earned Premium - Ceded Earned Premium(or Reinsurance costs))
net_prem_earned = {}
for i in range(1988,1998):
    net_prem_earned[i] = list( loss_data[loss_data['AccidentYear']==i]['EarnedPremNet_D'] )[0]
st.subheader("Earned Premium (Net)")
_df(net_prem_earned,"Earned Premium Net")

col10,col11,col12 = st.columns(3)

col10.subheader("Assume some rate changes (Other rate changes can be assumed).")

# Assume rate changes (for now, values are taken similar to those in Massachusetts Rate Filings)
rate_changes = {
            datetime.date(1988,4,1):0.198, #datetime.date(1989,1,1):0.1,
                datetime.date(1990,7,1):0.262, #datetime.date(1991,4,1):-0.04,
            datetime.date(1991,5,1):0.113,  #datetime.date(1992,3,1):0.07,
            datetime.date(1993,8,1):0.062, #datetime.date(1994,2,1):0.08,
            datetime.date(1996,5,1):-0.122
                }
rate_changes_df = pd.DataFrame({
    "rate change dates":rate_changes.keys(),
    "rate changes ":rate_changes.values()
})
col10.dataframe(rate_changes_df, hide_index=True)
# first calculate the rate change indeces
rates = list(rate_changes.values())
rate_index =[1.00]+[ (1+i) for i in rates ] # including initial index of segment without changes = 1.00 (rate change = 0%)
rate_index_df = pd.DataFrame({
    'rate index':rate_index
})
col11.dataframe(rate_index_df,hide_index=True)

cum_index = []
f = 1
for i in rate_index:
    f *= i
    cum_index.append( round(f, 4))
cum_index_df = pd.DataFrame({
    'cumulative rate index':cum_index
})
col12.dataframe(cum_index_df,hide_index=True)

current_cum_rate_index = cum_index[-1]
st.write("Current Cumulative Rate Level Index =",current_cum_rate_index)

# To calculate the portions earned by premiums under each rate change
T = 1; E = 1;

def months_between(date1,date2):
    '''This function calculates the difference between 2 given dates in months
    date1, date2 are in datetime.date() format'''
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2    # difference between the dates

    return months/12



def find_remains(rate_dates, earned_prem_year, L):
    '''This function calculates the remaining portions of earned premium under the rate changes
       rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
    if L!=[]:
        L.append(0) # appending 0 as a means for calculating the last portion
        to_return = []  # the list that contains the portions
        max = 1 # maximum value (total area of an year of earned premium)
        for i in range(0, len(L)):
            if L[i]!=0:
                diff = max - L[i]   # calculate remaining portion
                to_return.append(round( diff,5))
                max = L[i]
                if L[i+1]==0:   # for the last portion to be appended
                    to_return.append(round( max,5))
            else:
                to_return.append(0)

        if to_return.count(0) == len(to_return):
            to_return = earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, to_return)

        to_return.pop()
        return to_return



def earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, L):
    '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
    rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
    c = 0
    start_date = datetime.date(earned_prem_year,1,1)
    for i in rate_dates:
        if( months_between(i, start_date)>0 ):      # checking where to insert 1
            break
        else:
            c+=1
    L.insert(c,1)   # insert 1 as portion earned by premium
    return L



def earnedPortion(rate_dates, earned_prem_years):
    '''This function calculates the portion of earned premium under given rate changes
    rate_dates is a dictionary containing the dates of rate changes and earned_prem_years is also a dictionary containing the years in which premium is earned'''
    portion = {}
    for i in earned_prem_years:
        portion[i] = []
    for i in earned_prem_years:
        start_date = datetime.date(i,1,1)
        for j in rate_dates:

            if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
                # algorithm for calculating portions of earned premium
                D = months_between(j, start_date)

                A = D+T
                B = max( A-E, 0 )
                C = max( D, 0 )

                P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
                portion[i].append( round(P, 5))
            else:
                portion[i].append(0)

    for i in portion.keys():
        portion[i] = find_remains(rate_dates, i, portion[i])
    return(portion)


rate_effec_dates = list( rate_changes.keys())
years_toAdjust = list( net_prem_earned.keys() )
earned_NetPremPortion = earnedPortion(rate_effec_dates, years_toAdjust)
earned_NetPremPortion_df = pd.DataFrame({
    "Accident Years":earned_NetPremPortion.keys(),
    "Earned Premium (Net) Portions": earned_NetPremPortion.values()
})
st.subheader("The portion earned by the premium in the years w.r.t. the rate changes are:\n")
st.dataframe(earned_NetPremPortion_df,hide_index=True)

# Average Cumulative Rate Level Indices
def AvgCumulIndices(L, cumul_indices):
    '''This function calculates the average cumulative rate level indices for the earned premium
    L, cumul_indices are numpy arrays where L contains the portions of earned premiums and cumul_indices contains the cumulative rate level indices'''
    prod = L*cumul_indices
    sum = 0
    for i in prod:
        sum+=i
    return round(sum, 5)

# changing to numpy arrays
for i in earned_NetPremPortion.keys():
    earned_NetPremPortion[i] = np.array(earned_NetPremPortion[i])
cum_index = np.array(cum_index)

avg_CumulIndices = {}
for i in earned_NetPremPortion.keys():
    avg_CumulIndices[i] = AvgCumulIndices(earned_NetPremPortion[i], cum_index)
avg_CumulIndices_df = pd.DataFrame({
    "Accident Year": avg_CumulIndices.keys(),
    "Average Cumulative Rate Level Indices":avg_CumulIndices.values() 
})
st.subheader("Average Cumulative Rate level Indices for the respective accident years")
st.dataframe(avg_CumulIndices_df, hide_index=True)

col13, col14 = st.columns(2)
# On-Level Factors for the premiums
onlevel = {}
for i in avg_CumulIndices.keys():
    onlevel[i] = round( current_cum_rate_index/avg_CumulIndices[i], 5 )
onlevel_df = pd.DataFrame({
    "Accident Year": onlevel.keys(),
    "On-Level Factors":onlevel.values() 
})
col13.subheader("On-Level Factors for the premiums")
col13.dataframe(onlevel_df, hide_index=True)

# On-Levelling the Premiums
AdjustedPrem = {}
for i in onlevel.keys():
    AdjustedPrem[i] = round( net_prem_earned[i] * onlevel[i], 5)
AdjustedPrem_df = pd.DataFrame({
    "Accident Year": AdjustedPrem.keys(),
    "Rate Level Adjusted Premiums(Net)":AdjustedPrem.values() 
})
col14.subheader("Premiums adjusted for rate level changes:")
col14.dataframe(AdjustedPrem_df, hide_index=True)







"""## Adjusting Losses for Benefit Changes"""
col15,col16,col17 = st.columns(3)
col15.subheader("""Assume some benefit changes (Other benefit changes can be assumed).""")
# Assume benefit changes (Other benefit changes can be assumed)
benefit_changes = {
            # datetime.date(1988,4,1):0.05,
                 datetime.date(1989,1,1):0.1,
            # datetime.date(1990,7,1):-0.02,
                 datetime.date(1991,4,1):-0.04,
            # datetime.date(1991,5,1):0.11,
                 datetime.date(1992,3,1):0.07,
            # datetime.date(1993,8,1):-0.05,
                 datetime.date(1994,2,1):0.08,
            # datetime.date(1996,8,1):0.15
                }
benefit_changes_df = pd.DataFrame({
    "benefit change dates":benefit_changes.keys(),
    "benefit changes ":benefit_changes.values()
})
col15.dataframe(benefit_changes_df, hide_index=True)
# first calculate the benefit change indeces
benefits = list(benefit_changes.values())
benefit_index =[1.00]+[ (1+i) for i in benefits ] # including initial index without changes = 1.00 (rate change = 0%)
benefit_index_df = pd.DataFrame({
    "Benefit index":benefit_index
})
col16.dataframe(benefit_index_df, hide_index=True)

loss_lvl = []
f = 1
for i in benefit_index:
    f *= i
    loss_lvl.append( round(f, 4))
loss_lvl_df = pd.DataFrame({
    'loss level index':loss_lvl
})
col17.dataframe(loss_lvl_df,hide_index=True)

current_loss_lvl = loss_lvl[-1]
st.write("Current Loss Level Index =",current_loss_lvl)

T = 1; E = 1;

def months_between(date1,date2):
    '''This function calculates the difference between 2 given dates in months
    date1, date2 are in datetime.date() format'''
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2    # difference between the dates

    return months/12



def find_remains(ben_dates, loss_year, L):
    '''This function calculates the remaining portions of earned premium under the rate changes
       ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
    if L!=[]:
        L.append(0) # appending 0 as a means for calculating the last portion
        to_return = []  # the list that contains the portions
        max = 1 # maximum value (total area of an year of losses)
        for i in range(0, len(L)):
            if L[i]!=0:
                diff = max - L[i]   # calculate remaining portion
                to_return.append(round( diff,5))
                max = L[i]
                if L[i+1]==0:   # for the last portion to be appended
                    to_return.append(round( max,5))
            else:
                to_return.append(0)

        if to_return.count(0) == len(to_return):
            to_return = Portion_ForUnaffectedYear(ben_dates, loss_year, to_return)

        to_return.pop()
        return to_return



def Portion_ForUnaffectedYear(ben_dates, loss_year, L):
    '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
    ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
    c = 0
    start_date = datetime.date(loss_year,1,1)
    for i in ben_dates:
        if( months_between(i, start_date)>0 ):      # checking where to insert 1
            break
        else:
            c+=1
    L.insert(c,1)   # insert 1 as portion earned by premium
    return L



def LossPortion(ben_dates, loss_years):
    '''This function calculates the portion of earned premium under given rate changes
    ben_dates is a list containing the dates of benefit changes and loss_years is also a list containing the years in which losses occur'''
    portion = {}
    for i in loss_years:
        portion[i] = []
    for i in loss_years:
        start_date = datetime.date(i,1,1)
        for j in ben_dates:

            if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
                # algorithm for calculating portions of earned premium
                D = months_between(j, start_date)

                A = D+T
                B = max( A-E, 0 )
                C = max( D, 0 )

                P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
                portion[i].append( round(P, 5))
            else:
                portion[i].append(0)

    for i in portion.keys():
        portion[i] = find_remains(ben_dates, i, portion[i])
    return(portion)


ben_effec_dates = list( benefit_changes.keys())
years_toAdjust = list( ULT_LOSSES.keys() )
LossesPortion = LossPortion(ben_effec_dates, years_toAdjust)
LossesPortion_df = pd.DataFrame({
    "Accident Years":LossesPortion.keys(),
    "Loss Portions": LossesPortion.values()
})
st.subheader("The portion of the losses in the years w.r.t. the benefit changes are:\n")
st.dataframe(LossesPortion_df,hide_index=True)


# Average Loss Levels
def AvgLossLevel(L, loss_levels):
    '''This function calculates the average Loss levels for the historical periods
    L, loss_levels are numpy arrays where L contains the portions of losses and loss_levels contains the loss levels'''
    prod = L*loss_levels
    sum = 0
    for i in prod:
        sum+=i
    return round(sum, 5)

for i in LossesPortion.keys():
    LossesPortion[i] = np.array(LossesPortion[i])
loss_lvl = np.array(loss_lvl)

avg_LossLvl = {}
for i in LossesPortion.keys():
    avg_LossLvl[i] = AvgLossLevel(LossesPortion[i], loss_lvl)
avg_LossLvl_df = pd.DataFrame({
    "Accident Year": avg_LossLvl.keys(),
    "Average Loss Level Indices":avg_LossLvl.values() 
})
st.subheader("Average Loss level Indices for the respective accident years")
st.dataframe(avg_LossLvl_df, hide_index=True)

col18, col19 = st.columns(2)
# Adjustment Factors
adjusts = {}
st.write("Current Loss Level =",current_loss_lvl)
for i in avg_LossLvl.keys():
    adjusts[i] = round( current_loss_lvl/avg_LossLvl[i], 5 )
adjusts_df = pd.DataFrame({
    "Accident Year": adjusts.keys(),
    "Adjustment Factors":adjusts.values() 
})
col18.subheader("Adjustment Factors for the losses")
col18.dataframe(adjusts_df, hide_index=True)

# Adjusting the Losses
AdjustedLosses = {}
for i in adjusts.keys():
    AdjustedLosses[i] = round( ULT_LOSSES[i] * adjusts[i], 5)
AdjustedLosses_df = pd.DataFrame({
    "Accident Year": AdjustedLosses.keys(),
    "Benefit Adjusted Losses ":AdjustedLosses.values() 
})
col19.subheader("Losses adjusted for benefit changes:")
col19.dataframe(AdjustedLosses_df, hide_index=True)






"""# Trending Loss Ratios

### We are getting data related to Annual Inflation Rates by country from World Bank's website: [data.worldbank.org](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=US&view=chart)
"""

# Lets work on Inflation Rates first
filepath = "./605_InflationRates.csv"

# load the inflation dataset
inflation_rates = load_data(filepath)
inflation_rates

# inflation rates in USA
inf_us = inflation_rates[inflation_rates['Country Name'] == "United States"]
st.write("We extract the inflation rates for the USA:",inf_us)

inf_index = {}
start = 1988; end = 1997
for i in range(start, end+1):
    inf_index[i] = inf_us.iloc[0, i-1960+4]

st.write("Inflation indeces from 1988 to 1997 in the US are:\n")
_df(inf_index, "Inflation rate")

# average inflation rates
inf_avg = {}
periods = []
keys = list(inf_index.keys())
for i in range(0,len(keys)):
    avg=0
    temp = str(keys[i])+"-1997"
    for j in range(i,len(keys)):
        avg+= inf_index[keys[j]]
    periods.append(temp)
    inf_avg[keys[i]] = avg/(j-i+1)

st.write("The average inflation rates are:")
inf_avg_df = pd.DataFrame({
    "periods":periods,
    "average inflation":inf_avg.values()
})
st.dataframe(inf_avg_df,hide_index=True)


"""## Our Assumptions are:
### --> Policies are written uniformly over time.
### --> Premiums are earned uniformly over the policy period.
### --> Losses occur uniformly over the policy period.
### --> Policies have annual terms.

## Trend losses for inflation.
##### Our experience periods are the historical accident years from 1988 to 1997.
##### We assume future policy period begins on Jan 1, 1998 and inflation rate will be in effect for 12 months. Thus our forecast period average accident date is:
##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
"""
col20, col21, col22 = st.columns(3)
loss_inf_period = {}
loss_forecast_Date = datetime.date(1999,1,1)
for i in inf_index.keys():
    expDate = datetime.date(i,7,1)
    diff = months_between(loss_forecast_Date,expDate)
    loss_inf_period[i] = diff
col20.write("The trending periods for losses are:")
loss_inf_period_df = pd.DataFrame({
    "periods":periods,
    "trending periods":loss_inf_period.values()
})
col20.dataframe(loss_inf_period_df,hide_index=True)

# trend factors for losses
loss_inf_factor = {}
for i in loss_inf_period.keys():
    loss_inf_factor[i] = (1 + (0.01*inf_avg[i]))**loss_inf_period[i]
col21.write("The trend factors for losses are:")
loss_inf_factor_df = pd.DataFrame({
    "periods":periods,
    "trend factors":loss_inf_factor.values()
})
col21.dataframe(loss_inf_factor_df,hide_index=True)

# Now we trend the losses
inf_trendedLosses = {}
for i in loss_inf_factor.keys():
    inf_trendedLosses[i] = AdjustedLosses[i]*loss_inf_factor[i]
st.write("Losses trended for inflation:")
_df(inf_trendedLosses,"Inflation Trended Losses")



"""## Trend Premiums for inflation.
##### Trend will be estimated from earned premium data. The trend period will be from the average earned date in each historical period to the average earned date at the new rate level. Because of the uniform assumption, the average earned date of a period is the midpoint of the first and last dates that premiums could be earned in that period. So, these dates will depend on the policy term length.
##### Future policy period begins in Jan 1, 1998. Inflation rate will be in effect for 12 months. Thus our forecast period average earned date is:
##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
"""
col23, col24, col25 = st.columns(3)
prem_inf_period = {}
prem_forecast_Date = datetime.date(1999,1,1)
for i in inf_index.keys():
    expDate = datetime.date(i,1,1)
    diff = months_between(prem_forecast_Date,expDate)
    prem_inf_period[i] = diff
col23.write("The trending periods for premiums are:")
prem_inf_period_df = pd.DataFrame({
    "periods":periods,
    "trending periods":prem_inf_period.values()
})
col23.dataframe(prem_inf_period_df,hide_index=True)

# trend factors for premiums
prem_inf_factor = {}
for i in prem_inf_period.keys():
    prem_inf_factor[i] = (1 + (0.01*inf_avg[i]))**prem_inf_period[i]
col24.write("The trend factors for premiums are:")
prem_inf_factor_df = pd.DataFrame({
    "periods":periods,
    "trend factors":prem_inf_factor.values()
})
col24.dataframe(prem_inf_factor_df,hide_index=True)

# Now we trend the premiums
inf_trendedPrems = {}
for i in prem_inf_factor.keys():
    inf_trendedPrems[i] = AdjustedPrem[i]*prem_inf_factor[i]
st.write("Premiums trended for inflation:")
_df(inf_trendedPrems,"Inflation Trended Premiums")








"""# Expenses and Profits

## Assume fixed expense provision and variable expense provision. Also assume underwiting profit provision.
"""

fixed_exp_provision = 0.10       # 10%
variable_exp_provision = 0.15    # 15%
profit_provision = 0.015         # 1.5%

st.write("Fixed Expenses provision =",fixed_exp_provision*100,"%")
st.write("Variable Expenses provision =",variable_exp_provision*100,"%")
st.write("Target Underwriting Profit provision =",profit_provision*100,"%")

# permissible loss ratio
permissibleLR = 1 - (variable_exp_provision+profit_provision)
st.write("Permissible Loss Ratio = ", round(permissibleLR*100,3),"%")




"""# Overall Indicated Rate Change"""

# find the loss and lae ratios
loss_ratio = {}
for i in inf_trendedLosses.keys():
    loss_ratio[i] = inf_trendedLosses[i]/inf_trendedPrems[i]

# display the loss and lae ratios
loss_ratio_disp = {}
for i in loss_ratio.keys():
    loss_ratio_disp[i] = round(loss_ratio[i]*100, 3)
_df(loss_ratio_disp, "Loss(plus LAE) Ratio (in %)")

avg_loss_ratio = 0
for i in loss_ratio.keys():
    avg_loss_ratio+=loss_ratio[i]
avg_loss_ratio/=len(loss_ratio.keys())
st.write("Average loss ratio = ",round(avg_loss_ratio*100,2),"%")

if(avg_loss_ratio <= permissibleLR):
    st.write("Since, average loss ratio %.3f is less than permissible loss ratio %.3f,\nThe Company met underwriting profit expectations.\n"%(avg_loss_ratio,permissibleLR))
else :
    st.write("Since, average loss ratio %.3f is greater than permissible loss ratio %.3f,\nThe Company did not meet underwriting profit expectations.\n"%(avg_loss_ratio,permissibleLR))

# find overall rate level indicated change
indicated_avg_rate_change = ((avg_loss_ratio+fixed_exp_provision)/(1-variable_exp_provision-profit_provision)) - 1
st.write("Indicated average rate change for is=",round(indicated_avg_rate_change*100,4),"%")

