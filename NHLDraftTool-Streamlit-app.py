#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


# In[19]:


st.write("""
# National Hockey League (NHL) Draft Tool (Alpha)
### By Sean Farquharson

This app does not apply to top 10 ranked draft picks. 

This app predicts an NHL Draft prospects likelihood of playing in the NHL! 

This app is experimental. Several enhancements will be made to the underlying modeling procedure to make the tool more reliable.

## Quick User Guide

1) If uploading a csv file, please ensure column names are correct.

2) Calculations
- Player Size in Draft Year = Height(inches) in draft year + Weight(lbs) in draft year
- Goals Per Game = Goals scored/ Games Played
- Assists Per Game = Assists/ Games Played
- PIMs Per Game = PIMs/ Games Played

3) The application can be used for several functions including but not limited to:
- To build on to a player scouting profile.
- As a draft ranking tool, ranking by the predicted probability.
- Identifying underrated draft prospects.
- To gain insight into an ideal draft prospect by playing with the tool.
- To further distinguish between draft candidates for team scouting purposes, making a data-driven decision on draft selections.
- As a tool to aid in player development.

4) This tool may be updated or further improved upon in the future. Please send comments and/or inquiries to *sfarqu2@uwo.ca*.


Data was collected from NHL.com, Eliteprospects and thedraftanalyst.com. Obtained from https://github.com/liuyejia/Model_Trees_Full_Dataset

""")

st.sidebar.header('User Input Features')


# In[20]:


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Nationality = st.sidebar.selectbox('Nationality',('CAN','USA','EURO'))
        Position_in_Draft_Year = st.sidebar.selectbox('Position in Draft Year',('C','W','D'))
        Playoffs_in_Draft_Year = st.sidebar.selectbox('Team Made Playoffs in Draft Year?',('Yes','No'))
        Player_Size_in_Draft_Year = st.sidebar.slider('Player Size in Draft Year (Height(inches) + Weight(lbs))', 224.0, 344.0, 281.0)
        Goals_Per_Game_in_Draft_Year = st.sidebar.slider('Goals Per Game in Draft Year', 0.0, 1.75, 0.6)
        Assists_Per_Game_in_Draft_Year = st.sidebar.slider('Assists Per Game in Draft Year', 0.0, 2.2, 1.1)
        PIMs_Per_Game_in_Draft_Year = st.sidebar.slider('PIMs Per Game in Draft Year', 0.0, 8.0, 1.5)
        data = {'Nationality': Nationality,
           'PIMs_Per_Game_in_Draft_Year': PIMs_Per_Game_in_Draft_Year,
           'Goals_Per_Game_in_Draft_Year': Goals_Per_Game_in_Draft_Year,
           'Assists_Per_Game_in_Draft_Year': Assists_Per_Game_in_Draft_Year,
           'Player_Size_in_Draft_Year': Player_Size_in_Draft_Year,
           'Playoffs_in_Draft_Year': Playoffs_in_Draft_Year,
           'Position_in_Draft_Year': Position_in_Draft_Year}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# In[21]:


# Combine user input features with entire data set for encoding phase
NHL_raw = pd.read_csv('https://raw.githubusercontent.com/SeanFarquharson/NHL-Draft-Tool/main/NHLDraft_clean_for_web_app_streamlit.csv')
NHL = NHL_raw.drop(columns=['NHL_GP_Greater_Than_0'])
df = pd.concat([input_df,NHL],axis=0)


# In[22]:


# Encode
encode = ['Playoffs_in_Draft_Year','Position_in_Draft_Year','Nationality']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] #select only first row, the user input data


# In[23]:


# Display user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (see below).')
    st.write(df)


# In[24]:


# Read in saved RF model from pickle file
load_clf = pickle.load(open('NHLPicks_clf.pkl', 'rb'))


# In[25]:


# Apply model to predict
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
NHL_GP_Greater_Than_0 = np.array(['yes','no'])
st.write(NHL_GP_Greater_Than_0[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

