# %% [markdown]
# 
# # Latest API data
# This notebook consists of 3 parts:
# 1. Get new data from the API
# 2. Data preprocessing and feature engineering
# 3. Creating or backfilling the feature group

# %%
# Standard library imports for utilities and data manipulation
import uuid
import os
from datetime import datetime, timedelta
import io
import base64

# External libraries for data handling, networking, and time zones
import pandas as pd
import requests
import json
import pytz

# Environment variable management
from dotenv import load_dotenv
load_dotenv()

# Hopsworks
import hopsworks                

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
# Create a timezone object for GMT+2
timezone = pytz.timezone('Europe/Bucharest')
now = datetime.now(pytz.utc)  # Get current time 
today = now.astimezone(timezone)  # Convert to current timezone
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

# %%
# Format 'today', 'tomorrow', and 'yesterday' as "YYYY-MM-DD"
formatted_today = today.strftime('%Y-%m-%d')
formatted_tomorrow = tomorrow.strftime('%Y-%m-%d')
formatted_yesterday = yesterday.strftime('%Y-%m-%d')
url = "https://data.sensade.com"
dev_eui_building = "0080E115003BEA91"
dev_eui_bikelane = "0080E115003E3597"
username = "ajakup20@student.aau.dk"
basic_auth = base64.b64encode(f"{username}:{os.getenv('API_PASSWORD')}".encode())
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Basic {basic_auth.decode("utf-8")}'
}

# %%
# API call that takes the data from given sensor from yesterday until tomorrow
def API_call(dev_eui, from_date, to_date):
    payload = json.dumps({
    "dev_eui": dev_eui,
    "from": from_date,
    "to": to_date
})

    response = requests.request("GET", url, headers=headers, data=payload)

    if response.status_code != 200:
     print("Failed to fetch data: Status code", response.status_code)     
     print("Response:", response.text)     
     exit(13)

    csv_data = response.text
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

# %% [markdown]
# ## 2. Preprocessing and feature engineering
# 
# We apply the same methods as in notebook 1: creating unique IDs, converting the time column, converting radar names, changing data types to floats, and finally creating an empty column for the mag_cluster, as we haven't applied our models yet.

# %%
# Create a unique identifier for each row in the datasets
def create_id(df, dataset_name):
    # Assign the sensor prefix based on the dataset name
    if dataset_name == 'df_building_newest':
        df['psensor'] = "BUILDING"
    elif dataset_name == 'df_bikelane_newest':
        df['psensor'] = "BIKELANE"
    else:
        raise ValueError("Unknown dataset name provided")

    # Create a new column 'id' with a unique identifier for each row
    df['id'] = [str(uuid.uuid4()) for _ in df.index]

    return df

# %%
# Applying the function to the datasets
df_bikelane_newest = df_bikelane_newest.copy()
df_bikelane = create_id(df_bikelane_newest, 'df_bikelane_newest')
df_building_newest = df_building_newest.copy()
df_building = create_id(df_building_newest, 'df_building_newest')

# %%
#converting the time column to datetime
df_bikelane['time'] = pd.to_datetime(df_bikelane['time'])
df_building['time'] = pd.to_datetime(df_building['time'])

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
df_bikelane['mag_cluster'] = "null"
df_building['mag_cluster'] = "null"

# %% [markdown]
# ## Backfill or create feature group

# %%
# Connceting to the Hopsworks project

project = hopsworks.login()

fs = project.get_feature_store()

# %%
bikelane_fg = fs.get_or_create_feature_group(name="api_bikelane_newest",
                                  version=1,
                                  primary_key=["id"],
                                  event_time='time',
                                  description="New bike lane data",
                                  online_enabled=True,
                                 )
bikelane_fg.insert(df_bikelane)

# %%
building_fg = fs.get_or_create_feature_group(name="api_building_newest",
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


