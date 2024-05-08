# Loading packages
import datetime
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

# Configuring the web page and setting the page title and icon
st.set_page_config(
  page_title='Parking Occupacy Detection',
  page_icon='🅿️',
  initial_sidebar_state='expanded')

# Ignoring filtering warnings
warnings.filterwarnings("ignore")

# Setting the title and adding text
st.title('Parking Occupancy Detection')

# Creating tabs for the different features of the application
tab1,tab2,tab3,tab4, tab5 = st.tabs(['Parking lot status', 'Magnetic Field Explorer', 'About', 'Dataset and visualisations', 'Model performance'])

with tab1:
    # Logging in to Hopsworks and loading the feature store
    project = hopsworks.login(project = "miknie20", api_key_value=os.environ['HOPSWORKS_API_KEY'])
    fs = project.get_feature_store()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parking place near building:")

        # Function to load the building model
        
        @st.cache_data()
        def get_building_model(project=project):
            mr = project.get_model_registry()
            building_model = mr.get_model("building_hist_model", version = 2)
            building_model_dir = building_model.download()
            return joblib.load(building_model_dir + "/building_hist_model.pkl")

        # Retrieving model
        building_hist_model = get_building_model()
    
        # Loading the feature group with latest data for building
        api_building_newest_fg = fs.get_feature_group(name = 'api_building_newest', version = 1)

        # Function to loading the feature group with latest data for building as a dataset
        @st.cache_data()
        def retrieve_building(feature_group=api_building_newest_fg):
            api_building_newest_fg = feature_group.select(["time", "x", "y", "z"])
            df_building = api_building_newest_fg.read(read_options={"use_hive": True})           
            return df_building

        # Retrieving building data
        building_new = retrieve_building()

        # Making the predictions and getting the latest data
        building_most_recent_prediction = building_new[['x', 'y', 'z']]
        building_most_recent_prediction = building_hist_model.predict(building_most_recent_prediction)
        building_new['Status'] = building_most_recent_prediction
        building_new['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        building_new = building_new.rename(columns={'time': 'Time'})
        building_new = building_new.set_index(['Time'])
        st.dataframe(building_new[['Status']].tail(5))

    with col2:
        st.subheader("Parking place near bikelane:")
            
        # Function to load the bikelane model
        @st.cache_data()
        def get_bikelane_model(project=project):
            mr = project.get_model_registry()
            bikelane_model = mr.get_model("bikelane_hist_model", version = 1)
            bikelane_model_dir = bikelane_model.download()
            return joblib.load(bikelane_model_dir + "/bikelane_hist_model.pkl")

        # Retrieving model
        bikelane_hist_model = get_bikelane_model() 

        # Loading the feature group with latest data for bikelane
        api_bikelane_newest_fg = fs.get_feature_group(name = 'api_bikelane_newest', version = 1)

        # Function to loading the feature group with latest data for building as a dataset
        @st.cache_data()
        def retrieve_bikelane(feature_group=api_bikelane_newest_fg):
            api_bikelane_newest_fg = feature_group.select(["time", "x", "y", "z"])
            df_bikelane = api_bikelane_newest_fg.read(read_options={"use_hive": True})
            return df_bikelane

        # Retrieving building data
        bikelane_new = retrieve_bikelane()
        
        # Making the predictions and getting the latest data
        bikelane_most_recent_prediction = bikelane_new[['x', 'y', 'z']]
        bikelane_most_recent_prediction = bikelane_hist_model.predict(bikelane_most_recent_prediction)
        bikelane_new['Status'] = bikelane_most_recent_prediction
        bikelane_new['Status'].replace(['detection', 'no_detection'], ['Vehicle detected', 'No vehicle detected'], inplace=True)
        bikelane_new = bikelane_new.rename(columns={'time': 'Time'})
        bikelane_new = bikelane_new.set_index(['Time'])
        st.dataframe(bikelane_new[['Status']].tail(5)) 

    # Update button
    if st.button("Update application"):
        # Clear cached data
        st.cache_data.clear()
        # Immediately rerun the application
        st.experimental_rerun()
        
with tab2:
    # Defining a prediction function
    def explore_magnetic_field(model, x, y, z):
        input_list = [x, y, z]
        res = model.predict(np.asarray(input_list).reshape(1,-1))
        explorer_prediction = res[0]
        if explorer_prediction == 'detection':
            label = "Vehicle detected"
        else:
            label = "No vehicle detected"
        return label

    # Creating sliders for building model
    st.subheader('Experiment with building model:')
    x_input_building = st.slider("Choose your x-value", -232, 909, 0)
    y_input_building = st.slider("Choose your y-value", -1112, 435, 0)
    z_input_building = st.slider("Choose your z-value", -1648, 226, 0)

    # Making a prediction button for building model
    if st.button("Predict building input"):
        building_input_prediction = explore_magnetic_field(building_hist_model, x_input_building, y_input_building, z_input_building)
        st.write(building_input_prediction)
    
    st.divider()

    # Creating sliders for bikelane model
    st.subheader('Experiment with bikelane model:')
    x_input_bikelane = st.slider("Choose your x-value", -547, 288, 0)
    y_input_bikelane = st.slider("Choose your y-value", -1007, 786, 0)
    z_input_bikelane = st.slider("Choose your z-value", -1475, 16, 0)

    # Making a prediction button for bikelane model
    if st.button("Predict bikelane input"):
        bikelane_input_prediction = explore_magnetic_field(bikelane_hist_model, x_input_bikelane, y_input_bikelane, z_input_bikelane)
        st.write(bikelane_input_prediction)

with tab3:
    st.subheader('About the application:')
    st.markdown('This application is made as part of the module "Data Engineering and Machine Learning Operations in Business - F2024" in Business Data Science 2nd Semester at Aalborg University Business School.')
    st.markdown('The application is made by Annika and Mikkel and is divided into 5 tabs:')
    st.markdown('*   **Parking lot status:** The first tab includes the actual interface, where the goal has been to make a simple UI which shows if 2 parking spaces are occupied or available.')
    st.markdown('*   **Magnetic Field Explorer:** The second tabs is made for exploring the models, where the user can test different values for x, y and z and get a prediction')
    st.markdown('*   **About:** In the third tab (the current tab) you can get some information about the interface.')
    st.markdown('*   **Dataset and visualisations:** The fourth tab contains an overview of the training data and also includes EDAs for each individual parking space. The goal with these EDAs is to give you some information about when the parking spaces usually are occupied.')
    st.markdown('*   **Model Performance:** The fifth tab explains how the underlying Machine Learning Model performs and how the predictor works.')

with tab4:
    # Loading the feature group with historic data for building
    api_building_detection_features_fg = fs.get_feature_group(name = 'api_building_detection_features', version = 1)

    # Function to loading the feature group with latest data for building as a dataset
    @st.cache_data()
    def retrieve_building_historic(feature_group=api_building_detection_features_fg):
        api_building_detection_features_fg = feature_group.select_all()
        df_building_historic = api_building_detection_features_fg.read(read_options={"use_hive": True})           
        return df_building_historic

    # Retrieving building data
    building_historic = retrieve_building_historic()

    # Display historic building dataset overview
    st.subheader("Overview of the historic dataframe for the parking place near the building")
    st.dataframe(building_historic.head())

    st.markdown('Here we can see an overview of the columns in the historic dataframe for the building parking place. There is some missing data in the radar columns, but that does not affect our detection model, as it is built using the magnetic field data.')
    st.markdown('We have looked a bit into when there is most activity in the parking place. Looking at the two visualisations below, we can see that the parking place near the building has most changes between 03:00-06:00 and 10:00-15:00. We can also see that there is most activity on weekdays, which is as expected, as this parking place is outside an office.')    
    st.image('building_dist_hour.png')
    st.image('building_dist_week.png')

    st.markdown('The labels used to train the detection model are made on the basis of UML clustering. The clusters for the building parking place can be seen below.')
    st.image('building_cluster.png')
    
    st.divider()
    
    # Loading the feature group with historic data for bikelane
    api_bikelane_detection_features_fg = fs.get_feature_group(name = 'api_bikelane_detection_features', version = 1)

    # Function to loading the feature group with latest data for building as a dataset
    @st.cache_data()
    def retrieve_bikelane_historic(feature_group=api_bikelane_detection_features_fg):
        api_bikelane_detection_features_fg = feature_group.select_all()
        df_bikelane_historic = api_bikelane_detection_features_fg.read(read_options={"use_hive": True})           
        return df_bikelane_historic

    # Retrieving building data
    bikelane_historic = retrieve_bikelane_historic()

    # Display historic building dataset overview
    st.subheader("Overview of the historic dataset for the parking place near the bikelane")
    st.dataframe(bikelane_historic.head())  

    st.markdown('Looking at the historic dataframe for the bikelane parking place, we can see that it is quite similar to the one for the building parking place.')   
    st.markdown('There is a bit of difference in the hourly distribution of activity, as the parking place near the bikelane has most changes between 03:00-06:00 and 12:00-17:00. However, the weekly distribution shows the same.')    
    st.image('bikelane_dist_hour.png')
    st.image('bikelane_dist_week.png')

    st.markdown('Here we can see the clusters used to label the bikelane parking place.')
    st.image('bikelane_cluster.png')
    
with tab5:

    # Model performance of building model
    st.subheader('Model to predict parking place near building:')
    st.markdown('The predictions for the parking place near the building are made on the basis of a KNearestNeighbours model')
    st.write(building_hist_model)
    st.markdown('The accuracy if the bikelane model is 100%')
    st.write('**Confusion matrix:**')
    st.image('building_hist_confusion_matrix.png', caption='Confusion matrix for building model')

    st.divider()

    # Model performance of bikelane model
    st.subheader('Model to predict parking place near bikelane:')
    st.markdown('Just like with the other model, the predictions for the parking place near the bikelane are made on the basis of a KNearestNeighbours model')
    st.write(bikelane_hist_model)
    st.markdown('The accuracy if the building model is 99%')
    st.write('**Confusion matrix:**')
    st.image('bikelane_hist_confusion_matrix.png', caption='Confusion matrix for bikelane model')