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
project = hopsworks.login()
fs = project.get_feature_store()

# %%
# connect to the feature store
project = hopsworks.login()
fs = project.get_feature_store()

# %%
# Get the bike lane model from the model registry
mr = project.get_model_registry()
bikelane_model = mr.get_model("bikelane_hist_model", version=1)
model_dir = bikelane_model.download()
bikelane_model = joblib.load("/workspaces/MLOps_Project/notebooks/models/bikelane_hist_model.pkl")

# %%
# Get the building model from the model registry
building_model = mr.get_model("building_hist_model", version=2)
model_dir = building_model.download()
building_model = joblib.load("/workspaces/MLOps_Project/notebooks/models/building_hist_model.pkl")

# %%
# Retrieve the feature views for each parking spot
bikelane_feature_view = fs.get_feature_view(name="bikelane_new_fv", version=1)
building_feature_view = fs.get_feature_view(name="building_new_fv", version=1)

# %%
# Make predictions on the newest bikelane data
bikelane_data = bikelane_feature_view.get_batch_data()
bikelane_pred = bikelane_model.predict(bikelane_data)

# %%
# Make predictions on the newest building data
building_data = building_feature_view.get_batch_data()
building_pred = building_model.predict(building_data)

# %%
# Add the predictions to the dataframes
bikelane_data['mag_cluster'] = [bikelane_pred[bikelane_pred.size-1]]
building_data['mag_cluster'] = [building_pred[building_pred.size-1]]

# %%
#making a column describing if its data from bikelane or building
bikelane_data['data_type'] = ['bikelane']
building_data['data_type'] = ['building']

# %%
#combining the two dataframes
data = pd.concat([bikelane_data, building_data])

# %%
# upload the predictions to the feature store as a new feature group
latest_pred_fg = fs.get_or_create_feature_group(name="parking_predictions",
                                  version=1,
                                  primary_key=["x", "y", "z"],
                                  description="Predictions for parking spots",
                                  online_enabled=False,
                                 )
latest_pred_fg.insert(data)


