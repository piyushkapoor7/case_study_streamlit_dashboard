# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:55:10 2022

@author: Piyush Kapoor
"""

import scipy.stats as stats  
import numpy as np  
import pandas as pd 
import streamlit as st  
import matplotlib.pyplot as plt
import seaborn as sns



def data_load():
    data = pd.read_excel('Dashboarding case study_raw data.xlsx',skiprows=1)

    data = data[['article_id', 'article_name', 'sales_date', 'prediction_date',
       'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4',
       'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8',
       'prediction_9', 'prediction_10']]

    data_mean = data[['prediction_1', 'prediction_2', 'prediction_3', 'prediction_4',
       'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8',
       'prediction_9', 'prediction_10']]
    return data, data_mean

def main():
    data , data_mean = data_load()
    
    data['avg_pred'] = data_mean.mean(axis=1)
    data['avg_pred']=data['avg_pred'].apply(int)
    
    data_mean_flatten = data_mean.to_numpy().flatten()
    
    st.title('Sales Forecasting Dashboard')
    
    
    st.subheader('Distribution : All Predictions')
    fig4 = plt.figure()
    plt.hist(data_mean_flatten,color='purple',rwidth=0.7)
    st.pyplot(fig4)
    
    
    st.subheader('Day Wise Average Prediction')
    st.bar_chart(data,x='prediction_date',y='avg_pred',use_container_width=True)
    
    st.subheader('Overall Average')
    ov_mean = ['OverAll Meam',int(np.mean(np.mean(data_mean)))]
    st.table(ov_mean)
    
    st.subheader('Day Wise IQR View')
    date_wise = []
    confidence_interval_min = []
    confidence_interval_max = []
    day = []
    for i in data.index:
        ci_vals = []
        ind_col = 'Pred_Day_'+str(i)
        day.append(ind_col)
        for col in data_mean.columns:
            date_wise.append([ind_col,data[col][i]])
            ci_vals.append(data[col][i])
        confidence_interval = stats.t.interval(alpha=0.95, df=len(ci_vals)-1,loc=np.mean(ci_vals),scale=stats.sem(ci_vals))
        confidence_interval_min.append(int(confidence_interval[0]))
        confidence_interval_max.append(int(confidence_interval[1]))
    date_wise = pd.DataFrame(date_wise,columns=['Date','Prediction'])
    data['Day'] = day
    data['confidence_interval_min'] = confidence_interval_min
    data['confidence_interval_max'] = confidence_interval_max    
    fig = plt.figure()#figsize=(40, 15))
    fig.set_figwidth(15)
    fig.set_figheight(5)
    sns.boxplot(y=date_wise['Prediction'],x=date_wise['Date'])
    st.pyplot(fig)
    
    
    st.subheader('OverAll IQR View')
    fig2 = plt.figure()
    fig2.set_figwidth(15)
    fig2.set_figheight(5)
    sns.boxplot(y=date_wise['Prediction'])#,x=date_wise['Date'])
    st.pyplot(fig2)
    
    
    st.subheader('Day Wise Confidence Interval')
    fig3 = plt.figure()
    fig3.set_figwidth(15)
    fig3.set_figheight(5)
    plt.plot(data['Day'],data['avg_pred'],color='green')
    plt.fill_between(data['Day'],data['confidence_interval_min'], data['confidence_interval_max'], color='blue', alpha=0.1)
    st.pyplot(fig3)
    
    st.text("Confidence Interval 95%")
    
    st.table(data[['Day','prediction_date','avg_pred','confidence_interval_min','confidence_interval_max']])
    
    
    ci_ranges = [0.8,0.85,0.9,0.95]
    ci_all = []
    for ci in ci_ranges:
        ci_min_max  = stats.t.interval(alpha=ci, df=len(data_mean_flatten)-1,loc=np.mean(data_mean_flatten),scale=stats.sem(data_mean_flatten))
        ci_min = int(ci_min_max[0])
        ci_max = int(ci_min_max[1])
        ci_mn = np.mean(data_mean_flatten)
        tmp_lst = [ci,ci_mn,ci_min,ci_max]
        ci_all.append(tmp_lst)
    
    ci_all_df = pd.DataFrame(ci_all,columns=['Confidence Interval','Mean','Lower Limit','Upper Limit'])
    
    
    st.subheader('Confidence INterval on All Predictions')
    st.table(ci_all_df)


if __name__=="__main__":
    main()