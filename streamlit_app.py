import streamlit as st
import pandas as pd
import scipy
import sklearn

st.title('Fractal Analysis')

st.header('This is a web application for tumor fractal dimension')

st.file_uploader('', type=['jpeg', 'jpg', 'png'])
