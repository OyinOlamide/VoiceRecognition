import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as ff
from save_model import pickle
# from sklearn.linear_model import LogisticRegression



header = st.container()
dataset = st.container()
datavisual = st.container()
modelTraining = st.container()
footnote= st.container()

@st.cache
def getdata(filename):
    voice_data = pd.read_csv(r'C:\Users\ASHAROX\Downloads\voice.csv')
    return voice_data

with header:
    st.title('Gender Voice Recognition Classification')
    st.write('In this project I used various classifiers to blah blah blah for now')


with dataset:
    st.header('Voice dataset')
    st.text('I got this dataset from Kaggle')

    voice_data = getdata(r'C:\Users\ASHAROX\Downloads\voice.csv')

    if st.checkbox('Preview Dataset'):
        data = voice_data
        if st.button("HEAD"):
            st.write(data.head())
        elif st.button("TAIl"):
            st.write(data.tail())
        else:
            st.write(data.head(5))

with datavisual:
    st.header('Exploring the dataset to show visuals')

    if st.checkbox('Visuals'):
        if st.button('Show Histogram'):   
            fig = ff.histogram(voice_data,x='meanfreq',y='median',color='label')
            st.plotly_chart(fig, use_container_width=True)
            fig = ff.histogram(voice_data,x='sd',y='kurt',color='label')
            st.plotly_chart(fig, use_container_width=True)
        if st.button('Show Line chart'):
            df = pd.DataFrame(voice_data[:200],columns=['IQR','Q25','Q75'])
            st.line_chart(df)
        if st.button('Show Area Chart'):
            chart_data = pd.DataFrame(voice_data[:100],columns=['centroid','mode'])
            st.area_chart(chart_data)
        if st.button('Show Bar chart'):
            chart = pd.DataFrame(voice_data[:50],columns=['sfm','sp.ent'])
            st.bar_chart(chart)
            # st.bar_chart(voice_data[''])

with modelTraining:
    # st.header('Time to train the model')
    pickle_in = open('logit.pkl', 'rb')
    classifier = pickle.load(pickle_in)

    st.sidebar.header('Gender Recognition')
    select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    if not st.sidebar.checkbox("Hide", True, key='1'):
        st.title('Gender Recogition with voice features')
        
        meanfreq = st.number_input("Meanfreq value:")
        sd = st.number_input("sd value:")
        median = st.number_input("median value:")
        Q25 = st.number_input("Q25 value:")
        Q75 = st.number_input("Q75 value:")
        IQR = st.number_input("IQR value:")
        skew = st.number_input("skew value:")
        kurt = st.number_input("kurt value:")
        spent = st.number_input("sp.ent value:")
        sfm = st.number_input("sfm value")
        mode = st.number_input("sfm mode value:")
        centroid = st.number_input("centroid value:")
        meanfun = st.number_input("meanfun value:")
        minfun = st.number_input("minfun value:")
        maxfun = st.number_input("maxfun value:")
        meandom	= st.number_input("meandom value:")
        mindom = st.number_input("mindom value:")	
        maxdom = st.number_input("maxdom value:")
        dfrange = st.number_input("dfrange value:")
        modindx = st.number_input("modindx value:")
        
    submit = st.button('Predict')
    if submit:
            prediction = classifier.predict([[
                meanfreq,sd,median,Q25,Q75,IQR,skew,kurt,spent,sfm,mode,centroid,
                meanfun,minfun,maxfun,meandom,mindom,maxdom,dfrange,modindx]])
            if prediction == 0:
                st.write('Gender is a Female')
            else:
                st.write('Gender is a Male')
  
