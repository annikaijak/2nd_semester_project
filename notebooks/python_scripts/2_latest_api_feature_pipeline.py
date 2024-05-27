# %% [markdown]
# 
# # Latest API data
# This notebook consists of 3 parts:
# 1. Get new data from the API
# 2. Data preprocessing and feature engineering
# 3. Creating or backfilling the feature group

# %%
# Import standard Python libraries
import pandas as pd 
import hopsworks 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning tools
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score  

# Import other useful libraries
import uuid  # Unique identifier generation
import requests  # For making API requests
import json  
import io 
import os
import base64 
from datetime import datetime, timedelta  # Date/time handling and manipulation
import pytz  # Timezone conversions and support

import openmeteo_requests
import requests_cache
from retry_requests import retry

# Environment variable management
from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# ## 1. Get new data from the API
# 
# ### Sensor Data Access
# 
# Here is the information given by the company so we can acces thir data.
# 
# - GET request to `data.sensade.com` 
# - Authentication: `Basic Auth` (user: miknie20@student.aau.dk, password: GitHub Secret)
# 
# ### Sensors
# 
# Two sensors installed:
# 
# - `0080E115003BEA91` (Hw2.0 Fw2.0) Installed towards building
# 
# - `0080E115003E3597` (Hw2.0 Fw2.0) Installed towards bike lane

# %%
# getting the time for now
now = datetime.now()  # Get current time 
today = now 
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)
print(today)

# %%
# Format 'today', 'tomorrow', and 'yesterday' as "YYYY-MM-DD"
formatted_today = today.strftime('%Y-%m-%d %H:%M:%S')
formatted_tomorrow = tomorrow.strftime('%Y-%m-%d %H:%M:%S')
formatted_yesterday = yesterday.strftime('%Y-%m-%d %H:%M:%S')

# %%
# Defining API information
dev_eui_building = "0080E115003BEA91"
dev_eui_bikelane = "0080E115003E3597"
url = "https://data.sensade.com"

basic_auth = base64.b64encode(f"{os.getenv('API_USERNAME')}:{os.getenv('API_PASSWORD')}".encode())
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Basic {basic_auth.decode("utf-8")}'
}

# %%
# Function to ping the API and get data in a given time interval
def API_call(dev_eui, from_date, to_date):
    payload = json.dumps({
    "dev_eui": dev_eui,
    "from": from_date,
    "to": to_date
})

    API_response = requests.request("GET", url, headers=headers, data=payload)

    if API_response.status_code != 200:
        exit(13)

    csv_data = API_response.text
    df = pd.read_csv(io.StringIO(csv_data))
    return df

# %%
# Running the API call function with the given parameters on the building sensor
df_building_from_api = API_call(dev_eui_building, formatted_yesterday, formatted_tomorrow)

# %%
# Running the API call function with the given parameters on the bikelane sensor
df_bikelane_from_api = API_call(dev_eui_bikelane, formatted_yesterday, formatted_tomorrow)

# %%
# Defning the newest data from the API calls
df_building_newest = df_building_from_api.tail(1)
df_bikelane_newest = df_bikelane_from_api.tail(1)

# %%
df_building_newest

# %%
df_bikelane_newest

# %% [markdown]
# ## 2. Preprocessing and feature engineering
# 
# We apply the same methods as in notebook 1: creating unique IDs, converting the time column, converting radar names, changing data types to floats, and finally creating an empty column for the mag_cluster, as we haven't applied our models yet.

# %%
df_building = df_building_newest.copy()
df_bikelane = df_bikelane_newest.copy()

# %%
# Defining a function that tries to parse the datetime with microseconds first, and if it fails, parses it without microseconds
def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

# %%
# Applying the function on the dataframes
df_building = df_building.copy()
df_building['time'] = df_building['time'].apply(parse_datetime)
df_bikelane = df_bikelane.copy()
df_bikelane['time'] = df_bikelane['time'].apply(parse_datetime)

# %%
#converting the time column to datetime
df_bikelane['time'] = pd.to_datetime(df_bikelane['time'])
df_building['time'] = pd.to_datetime(df_building['time'])

# %%
#create a column for the time in the format of "YYYY-MM-DD HH" to merge with weather data
df_bikelane['time_hour'] = df_bikelane['time'].dt.strftime('%Y-%m-%d %H')
df_building['time_hour'] = df_building['time'].dt.strftime('%Y-%m-%d %H')
# Converting the time_hour column to datetime
df_bikelane['time_hour'] = pd.to_datetime(df_bikelane['time_hour'])
df_building['time_hour'] = pd.to_datetime(df_building['time_hour'])

# %% [markdown]
# ### Weather data column

# %%
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# %%

weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
	"latitude": 57.01,
	"longitude": 9.99,
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "cloud_cover", "et0_fao_evapotranspiration", "wind_speed_10m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"],
	"forecast_days": 1
}
responses = openmeteo.weather_api(weather_url, params=weather_params)

# %%
# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
hourly_et0_fao_evapotranspiration = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_soil_temperature_0_to_7cm = hourly.Variables(7).ValuesAsNumpy()
hourly_soil_moisture_0_to_7cm = hourly.Variables(8).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.head()

# %%
#remove the timezone from the date column
hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_localize(None)
#Convert to datetime object
hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])

# %% [markdown]
# # Merging weather data and sensor data

# %%
# Merging the weather data with the building sensor data
df_building = pd.merge(df_building, hourly_dataframe, left_on='time_hour', right_on='date', how='left')
# Merging the weather data with the bikelane sensor data
df_bikelane = pd.merge(df_bikelane, hourly_dataframe, left_on='time_hour', right_on='date', how='left')

# %%
df_building = df_building.drop(columns=['date'])
df_bikelane = df_bikelane.drop(columns=['date'])

# %% [markdown]
# ## Feature Engineering

# %%
#Adding two hours to datetime to match the timezone
df_bikelane['time'] = df_bikelane['time'] + pd.Timedelta(hours=2)
df_building['time'] = df_building['time'] + pd.Timedelta(hours=2)

# %%
# Create a unique identifier for each row in the datasets
def create_id(df, dataset_name):
    # Assign the sensor prefix based on the dataset name
    if dataset_name == 'df_building':
        df['psensor'] = "BUILDING"
    elif dataset_name == 'df_bikelane':
        df['psensor'] = "BIKELANE"
    else:
        raise ValueError("Unknown dataset name provided")

    # Create a new column 'id' with a unique identifier for each row
    df['id'] = df['time'].astype(str) + '_' + df['psensor']

    return df

# %%
# Applying the function to the datasets
df_bikelane = create_id(df_bikelane, 'df_bikelane')
df_building = create_id(df_building, 'df_building')

# %%
#Renaming the radar columns to start with radar
df_bikelane = df_bikelane.rename(columns={'0_radar': 'radar_0', '1_radar': 'radar_1', '2_radar': 'radar_2', '3_radar': 'radar_3', '4_radar': 'radar_4', '5_radar': 'radar_5', '6_radar': 'radar_6', '7_radar': 'radar_7'})
df_building = df_building.rename(columns={'0_radar': 'radar_0', '1_radar': 'radar_1', '2_radar': 'radar_2', '3_radar': 'radar_3', '4_radar': 'radar_4', '5_radar': 'radar_5', '6_radar': 'radar_6', '7_radar': 'radar_7'})

# %%
# Converting the columns to float
df_bikelane[['x','y','z', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'f_cnt', 'dr', 'rssi']] = df_bikelane[['x','y','z', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'f_cnt', 'dr', 'rssi']].astype(float)
df_building[['x','y','z', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'f_cnt', 'dr', 'rssi']] = df_building[['x','y','z', 'radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7', 'f_cnt', 'dr', 'rssi']].astype(float)


# %%
#making an empty label column
df_bikelane['radar_cluster'] = "null"
df_building['radar_cluster'] = "null"
df_bikelane['mag_cluster'] = "null"
df_building['mag_cluster'] = "null"

# %% [markdown]
# ## Uploading latest data to Hopsworks

# %%
# Connceting to the Hopsworks project

project = hopsworks.login(project="annikaij")

fs = project.get_feature_store()

# %%
bikelane_fg = fs.get_or_create_feature_group(name="new_bikelane_fg",
                                  version=1,
                                  primary_key=["id"],
                                  event_time='time',
                                  description="New bike lane data",
                                  online_enabled=True,
                                 )
bikelane_fg.insert(df_bikelane)

# %%
building_fg = fs.get_or_create_feature_group(name="new_building_fg",
                                    version=1,
                                    primary_key=["id"],
                                    event_time='time',
                                    description="New building data",
                                    online_enabled=True
                                     )
building_fg.insert(df_building)

# %% [markdown]
# ## **Next up:** 3: Feature view creation
# Go to the 3_featureview_creation.ipynb notebook


