# Loading packages
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import hopsworks
import streamlit as st
import json
import os
import seaborn as sns
import time
import random
from sklearn.preprocessing import StandardScaler

# Configuring the web page and setting the page title and icon
st.set_page_config(
  page_title='Parking Occupacy Detection',
  page_icon='🅿️',
  initial_sidebar_state='expanded')

# Ignoring filtering warnings
warnings.filterwarnings("ignore")

# Setting the title and adding text
st.title('Parking Occupancy Detection')

# Defining functions
def fill_nan_with_zero(value):
    if pd.isna(value):
        return 0
    else:
        return value

# Getting current time and yesterday
now = datetime.now() + timedelta(hours=2)
yesterday = now - timedelta(days=1)

# Defining scaler
scaler = StandardScaler()

# Creating tabs for the different features of the application
tab1,tab2 = st.tabs(['Parking place near Building', 'Parking place near Bikelane'])

with tab1:
    # Logging in to Hopsworks and loading the feature store
    project = hopsworks.login(project = "annikaij", api_key_value=os.environ['HOPSWORKS_API_KEY'])
    fs = project.get_feature_store()

    # Function to load the building models
        
    @st.cache_data()
    def get_building_mag_model(project=project):
        mr = project.get_model_registry()
        building_mag_model = mr.get_model("building_mag_hist_model", version = 2)
        building_mag_model_dir = building_mag_model.download()
        return joblib.load(building_mag_model_dir + "/building_mag_hist_model.pkl")

    # Retrieving model
    building_mag_hist_model = get_building_mag_model()

    @st.cache_data()
    def get_building_rad_model(project=project):
        mr = project.get_model_registry()
        building_rad_model = mr.get_model("building_rad_hist_model", version = 2)
        building_rad_model_dir = building_rad_model.download()
        return joblib.load(building_rad_model_dir + "/building_rad_hist_model.pkl")

    # Retrieving model
    building_rad_hist_model = get_building_rad_model()
    
    # Loading the feature group with latest data for building
    new_building_fg = fs.get_feature_group(name = 'new_building_fg', version = 1)

    # Function to loading the feature group with latest data for building as a dataset
    @st.cache_data()
    def retrieve_building(feature_group=new_building_fg):
        new_building_fg = feature_group.select_all()
        df_building_new = new_building_fg.read(read_options={"use_hive": True})           
        return df_building_new

    # Retrieving building data
    building_new = retrieve_building()
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Magnetic field prediction")
        
        # Making the predictions and getting the latest data for magnetic field data
        building_mag_prediction_data = building_new[['time', 'x', 'y', 'z', 'temperature', 'et0_fao_evapotranspiration']]                       
        building_mag_prediction_data['et0_fao_evapotranspiration'] = building_mag_prediction_data['et0_fao_evapotranspiration'].apply(fill_nan_with_zero)
        building_mag_most_recent_prediction = building_mag_prediction_data[['x', 'y', 'z', 'temperature', 'et0_fao_evapotranspiration']]
        building_mag_most_recent_prediction = building_mag_hist_model.predict(building_mag_most_recent_prediction)
        building_mag_prediction_data['Status'] = building_mag_most_recent_prediction
        building_mag_prediction_data['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        building_mag_prediction_data = building_mag_prediction_data.rename(columns={'time': 'Time'})
        building_mag_prediction_data = building_mag_prediction_data.set_index(['Time'])
        st.dataframe(building_mag_prediction_data[['Status']].tail(3))

    with col2:
        st.subheader("Radar prediction")
        
        # Making the predictions and getting the latest data for radar data
        building_rad_prediction_data = building_new[['time', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'temperature', 'et0_fao_evapotranspiration']]                       
        building_rad_prediction_data['et0_fao_evapotranspiration'] = building_rad_prediction_data['et0_fao_evapotranspiration'].apply(fill_nan_with_zero)
        building_rad_most_recent_prediction = building_rad_prediction_data[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'temperature', 'et0_fao_evapotranspiration']]
        building_rad_most_recent_prediction = building_rad_hist_model.predict(building_rad_most_recent_prediction)
        building_rad_prediction_data['Status'] = building_rad_most_recent_prediction
        building_rad_prediction_data['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        building_rad_prediction_data = building_rad_prediction_data.rename(columns={'time': 'Time'})
        building_rad_prediction_data = building_rad_prediction_data.set_index(['Time'])
        st.dataframe(building_rad_prediction_data[['Status']].tail(3))

    # Update button
    if st.button("Update Building"):
        # Clear cached data
        st.cache_data.clear()
        # Immediately rerun the application
        st.experimental_rerun()

    # Creating plot for latest magnetic field data for building
    # Filtering building_new for specific time
    building_mag_specific_time_range = building_new[(building_new['time'] >= yesterday) & (building_new['time'] <= now)]

    # Defining magnetic field data to normalise
    building_mag_to_normalize = building_mag_specific_time_range[['x', 'y', 'z']]

    # Applying StandardScaler
    normalized_building_mag = scaler.fit_transform(building_mag_to_normalize)    

    # Adding normalized data back to the DataFrame
    building_mag_specific_time_range[['x', 'y', 'z']] = normalized_building_mag

    # Streamlit plotting
    st.subheader('Normalized values of magnetic field data from yesterday to today')

    # Converting the time column to string for better readability in Streamlit plots
    building_mag_specific_time_range['time'] = building_mag_specific_time_range['time'].astype(str)

    # Plotting using Streamlit's line chart
    st.line_chart(building_mag_specific_time_range.set_index('time')[['x', 'y', 'z']])

    # Creating plot for latest radar data for building
    # Filtering building_new for specific time
    building_rad_specific_time_range = building_new[(building_new['time'] >= yesterday) & (building_new['time'] <= now)]

    # Defining magnetic field data to normalise
    building_rad_to_normalize = building_rad_specific_time_range[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']]

    # Applying StandardScaler
    normalized_building_rad = scaler.fit_transform(building_rad_to_normalize)    

    # Adding normalized data back to the DataFrame
    building_rad_specific_time_range[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']] = normalized_building_rad

    # Streamlit plotting
    st.subheader('Normalized values of radar data from yesterday to today')

    # Converting the time column to string for better readability in Streamlit plots
    building_rad_specific_time_range['time'] = building_rad_specific_time_range['time'].astype(str)

    # Plotting using Streamlit's line chart
    st.line_chart(building_rad_specific_time_range.set_index('time')[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']])
        
with tab2:

    # Function to load the bikelane models
        
    @st.cache_data()
    def get_bikelane_mag_model(project=project):
        mr = project.get_model_registry()
        bikelane_mag_model = mr.get_model("bikelane_mag_hist_model", version = 2)
        bikelane_mag_model_dir = bikelane_mag_model.download()
        return joblib.load(bikelane_mag_model_dir + "/bikelane_mag_hist_model.pkl")

    # Retrieving model
    bikelane_mag_hist_model = get_bikelane_mag_model()

    @st.cache_data()
    def get_bikelane_rad_model(project=project):
        mr = project.get_model_registry()
        bikelane_rad_model = mr.get_model("bikelane_rad_hist_model", version = 2)
        bikelane_rad_model_dir = bikelane_rad_model.download()
        return joblib.load(bikelane_rad_model_dir + "/bikelane_rad_hist_model.pkl")

    # Retrieving model
    bikelane_rad_hist_model = get_bikelane_rad_model()
    
    # Loading the feature group with latest data for bikelane
    new_bikelane_fg = fs.get_feature_group(name = 'new_bikelane_fg', version = 1)

    # Function to loading the feature group with latest data for bikelane as a dataset
    @st.cache_data()
    def retrieve_bikelane(feature_group=new_bikelane_fg):
        new_bikelane_fg = feature_group.select_all()
        df_bikelane_new = new_bikelane_fg.read(read_options={"use_hive": True})           
        return df_bikelane_new

    # Retrieving bikelane data
    bikelane_new = retrieve_bikelane()

    col1, col2 = st.columns(2)

    with col1:    
        st.subheader("Magnetic field prediction")
        # Making the predictions and getting the latest data for magnetic field data
        bikelane_mag_prediction_data = bikelane_new[['time', 'x', 'y', 'z', 'temperature', 'et0_fao_evapotranspiration']]                       
        bikelane_mag_prediction_data['et0_fao_evapotranspiration'] = bikelane_mag_prediction_data['et0_fao_evapotranspiration'].apply(fill_nan_with_zero)
        bikelane_mag_most_recent_prediction = bikelane_mag_prediction_data[['x', 'y', 'z', 'temperature', 'et0_fao_evapotranspiration']]
        bikelane_mag_most_recent_prediction = bikelane_mag_hist_model.predict(bikelane_mag_most_recent_prediction)
        bikelane_mag_prediction_data['Status'] = bikelane_mag_most_recent_prediction
        bikelane_mag_prediction_data['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        bikelane_mag_prediction_data = bikelane_mag_prediction_data.rename(columns={'time': 'Time'})
        bikelane_mag_prediction_data = bikelane_mag_prediction_data.set_index(['Time'])
        st.dataframe(bikelane_mag_prediction_data[['Status']].tail(3))

    with col2:  
        st.subheader("Radar prediction")
        # Making the predictions and getting the latest data for radar data
        bikelane_rad_prediction_data = bikelane_new[['time', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'temperature', 'et0_fao_evapotranspiration']]                       
        bikelane_rad_prediction_data['et0_fao_evapotranspiration'] = bikelane_rad_prediction_data['et0_fao_evapotranspiration'].apply(fill_nan_with_zero)
        bikelane_rad_most_recent_prediction = bikelane_rad_prediction_data[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'temperature', 'et0_fao_evapotranspiration']]
        bikelane_rad_most_recent_prediction = bikelane_rad_hist_model.predict(bikelane_rad_most_recent_prediction)
        bikelane_rad_prediction_data['Status'] = bikelane_rad_most_recent_prediction
        bikelane_rad_prediction_data['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        bikelane_rad_prediction_data = bikelane_rad_prediction_data.rename(columns={'time': 'Time'})
        bikelane_rad_prediction_data = bikelane_rad_prediction_data.set_index(['Time'])
        st.dataframe(bikelane_rad_prediction_data[['Status']].tail(3))

    # Update button
    if st.button("Update Bikelane"):
        # Clear cached data
        st.cache_data.clear()
        # Immediately rerun the application
        st.experimental_rerun()

    # Creating plot for latest magnetic field data for bikelane
    # Filtering bikelane_new for specific time
    bikelane_mag_specific_time_range = bikelane_new[(bikelane_new['time'] >= yesterday) & (bikelane_new['time'] <= now)]

    # Defining magnetic field data to normalise
    bikelane_mag_to_normalize = bikelane_mag_specific_time_range[['x', 'y', 'z']]

    # Applying StandardScaler
    normalized_bikelane_mag = scaler.fit_transform(bikelane_mag_to_normalize)    

    # Adding normalized data back to the DataFrame
    bikelane_mag_specific_time_range[['x', 'y', 'z']] = normalized_bikelane_mag

    # Streamlit plotting
    st.subheader('Normalized values of magnetic field data from yesterday to today')

    # Converting the time column to string for better readability in Streamlit plots
    bikelane_mag_specific_time_range['time'] = bikelane_mag_specific_time_range['time'].astype(str)

    # Plotting using Streamlit's line chart
    st.line_chart(bikelane_mag_specific_time_range.set_index('time')[['x', 'y', 'z']])

    
    # Creating plot for latest radar data for bikelane
    # Filtering bikelane_new for specific time
    bikelane_rad_specific_time_range = bikelane_new[(bikelane_new['time'] >= yesterday) & (bikelane_new['time'] <= now)]

    # Defining magnetic field data to normalise
    bikelane_rad_to_normalize = bikelane_rad_specific_time_range[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']]

    # Applying StandardScaler
    normalized_bikelane_rad = scaler.fit_transform(bikelane_rad_to_normalize)    

    # Adding normalized data back to the DataFrame
    bikelane_rad_specific_time_range[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']] = normalized_bikelane_rad

    # Streamlit plotting
    st.subheader('Normalized values of radar data from yesterday to today')

    # Converting the time column to string for better readability in Streamlit plots
    bikelane_rad_specific_time_range['time'] = bikelane_rad_specific_time_range['time'].astype(str)

    # Plotting using Streamlit's line chart
    st.line_chart(bikelane_rad_specific_time_range.set_index('time')[['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7']])