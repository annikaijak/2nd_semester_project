# %% [markdown]
# # 5. Inference pipeline
# This notebook consists of 3 parts:
# 
# 1. Connecting to the Feature Store and retriving feature views/groups and model
# 3. Predicting if theres a detection or not
# 4. Saving the prediction in a new feature view/group

# %%
# import libraries
import pandas as pd
import hopsworks
import joblib

# %% [markdown]
# ## 1. Connecting to the Feature Store and retriving feature views/groups and model

# %%
project = hopsworks.login(project="annikaij")
fs = project.get_feature_store()

# %%
# connect to the feature store
project = hopsworks.login(project="annikaij")
fs = project.get_feature_store()

# %%
# Get the magnetic bike lane model from the model registry
mr = project.get_model_registry()
mag_bikelane_model = mr.get_model("bikelane_mag_hist_model", version=2)
model_dir = mag_bikelane_model.download()
mag_bikelane_model = joblib.load("/workspaces/2nd_semester_project/notebooks/models/bikelane_mag_hist_model.pkl")

# %%
# Get the magnetic building model from the model registry
mag_building_model = mr.get_model("building_mag_hist_model", version=2)
model_dir = mag_building_model.download()
mag_building_model = joblib.load("/workspaces/2nd_semester_project/notebooks/models/building_mag_hist_model.pkl")

# %%
# get the radar bikelane model from the model registry
radar_bikelane_model = mr.get_model("bikelane_rad_hist_model", version=2)
model_dir = radar_bikelane_model.download()
radar_bikelane_model = joblib.load("/workspaces/2nd_semester_project/notebooks/models/bikelane_rad_hist_model.pkl")

# %%
# get the radar building model from the model registry
radar_building_model = mr.get_model("building_rad_hist_model", version=2)
model_dir = radar_building_model.download()
radar_building_model = joblib.load("/workspaces/2nd_semester_project/notebooks/models/building_rad_hist_model.pkl")

# %%
# Retrieve the feature views for each parking spot
mag_bikelane_feature_view = fs.get_feature_view(name="hist_bikelane_mag_fv", version=1)
mag_building_feature_view = fs.get_feature_view(name="hist_building_mag_fv", version=1)
rad_bikelane_feature_view = fs.get_feature_view(name="hist_bikelane_radar_fv", version=1)
rad_building_feature_view = fs.get_feature_view(name="hist_building_radar_fv", version=1)


# %%
# Make predictions on the newest magnetic bikelane data
mag_bikelane_data = mag_bikelane_feature_view.get_batch_data()
mag_bikelane_pred = mag_bikelane_model.predict(mag_bikelane_data)

# %%
# Make predictions on the newest magnetic building data
mag_building_data = mag_building_feature_view.get_batch_data()
mag_building_pred = mag_building_model.predict(mag_building_data)


# %%
# Make predictions on the newest radar bikelane data
rad_bikelane_data = rad_bikelane_feature_view.get_batch_data()
rad_bikelane_pred = radar_bikelane_model.predict(rad_bikelane_data)

# %%
# Make predictions on the newest radar building data
rad_building_data = rad_building_feature_view.get_batch_data()
rad_building_pred = radar_building_model.predict(rad_building_data)

# %%
# Add the predictions to the dataframes
mag_bikelane_data['mag_cluster'] = mag_bikelane_pred[mag_bikelane_pred.size-1]
mag_building_data['mag_cluster'] = mag_building_pred[mag_building_pred.size-1]
rad_bikelane_data['rad_cluster'] = rad_bikelane_pred[rad_bikelane_pred.size-1]
rad_building_data['rad_cluster'] = rad_building_pred[rad_building_pred.size-1]


# %%
#making a column describing if its data from bikelane or building
mag_bikelane_data['data_type'] = 'bikelane'
mag_building_data['data_type'] = 'building'
rad_bikelane_data['data_type'] = 'bikelane'
rad_building_data['data_type'] = 'building'

# %%
#combining the two dataframes
mag_data = pd.concat([mag_bikelane_data, mag_building_data])
rad_data = pd.concat([rad_bikelane_data, rad_building_data])

# %%
# upload the predictions to the feature store as a new feature group
latest_pred_fg = fs.get_or_create_feature_group(name="mag_parking_predictions",
                                  version=1,
                                  primary_key=["x", "y", "z"],
                                  description="Predictions for parking spots with magnetic data",
                                  online_enabled=False,
                                 )
latest_pred_fg.insert(mag_data)

# %%
# upload the radar predictions to the feature store as a new feature group
latest_pred_fg = fs.get_or_create_feature_group(name="rad_parking_predictions",
                                  version=1,
                                  primary_key=['radar_0', 'radar_1', 'radar_2', 'radar_3', 'radar_4', 'radar_5', 'radar_6', 'radar_7'],
                                  description="Predictions for parking spots with radar data",
                                  online_enabled=False,
                                 )
latest_pred_fg.insert(rad_data)


