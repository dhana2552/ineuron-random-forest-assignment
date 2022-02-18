import streamlit as st
import pickle
import sklearn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

st.title('Random Forest Assignment')
st.info('It is expected to predict boston house price median value in the selected region using Random Forest Regressor algorithms. '
        'The prediction was made with all the features and only few are made available for end users')

#CHAS
chas = st.selectbox('Select Charles River dummy variable', df['CHAS'].unique())

#ZN
zn = st.selectbox('Select the proportion of residential land zoned', df['ZN'].unique())

#RAD
rad = st.selectbox('Select the index of accessibility to radial highways', df['RAD'].unique())

#CRIM
crim = st.number_input('Enter per capita crime rate by town (0.00 to 100.00)')

#INDUS
indus = st.number_input('Enter proportion of non-retail business acres per town (0.00 to 30.00)')

#AGE
age = st.number_input('Enter proportion of owner-occupied units built prior to 1940 (0.00 to 100.00)')

nox = np.random.choice(df.NOX)
rm = np.random.choice(df.RM)
dis = np.random.choice(df.DIS)
tax = np.random.choice(df.TAX)
pt_ratio = np.random.choice(df.PTRATIO)
b = np.random.choice(df.B)
lstat = np.random.choice(df.LSTAT)

if st.button('Predict'):
    #CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
    query = np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, pt_ratio, b, lstat]).reshape(1, 13)
    st.title(model.predict(query)[0])
    st.markdown("PS: This is the predicted Median value of owner-occupied homes in $1000's.\n"
                "\nThe other feature values that resulted in this predicted median are as follows:")
    st.info(f"NOX={nox}, RM={rm}, DIS={dis}, TAX={tax}, PT_RATIO={pt_ratio}, B={b}, LSTAT={lstat}")
