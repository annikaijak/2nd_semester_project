# %% [markdown]
# # Model training
# This notebook consists of 4 parts:
# 
# 1. Connecting to the Feature Store
# 2. Creating training data with the newest data in the "historic" feature views
# 3. Training models for each parking spot
# 4. Uploading/updating models and their performance in the Feature Store and Github

# %%
# Standard library imports
import os
import shutil
import joblib

# Data handling and analysis
import pandas as pd

# Visualization libraries
from matplotlib import pyplot
import seaborn as sns

# Machine Learning: model and metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hopsworks-related imports
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


# %% [markdown]
# ## 1. Connecting to the Feature Store

# %%
project = hopsworks.login(project="annikaij")
fs = project.get_feature_store()

# %% [markdown]
# ## 2. create training data

# %%
# Get the latest version of the magnetic field data feature view for bikelane
version=1
feature_view_mag_bikelane = fs.get_feature_view("hist_bikelane_mag_fv", version=version)

# %%
# Get the latest version of the training dataset
X_train, X_test, y_train, y_test = feature_view_mag_bikelane.train_test_split(0.2)

# %%
# Check the distribution of the target variable
y_train.value_counts(normalize=True)

# %% [markdown]
# ## 3. Train, test and evaluate model for bikelane magnetic data

# %%
# Define the model
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())

# %%
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred

# %%
# make classification report
metrics = classification_report(y_test, y_pred, output_dict=True)
metrics

# %%
# save the results in the model schema
results = confusion_matrix(y_test, y_pred)

# %%
# Make and save the confusion matrix
df_cm = pd.DataFrame(results, ['True Detection', 'True no Detection'],
                     ['Pred Detection', 'Pred no Detection'])

cm = sns.heatmap(df_cm, annot=True, fmt='g')

fig = cm.get_figure()
fig.savefig("/workspaces/2nd_semester_project/pictures/knn_mag_bikelane_matrix.png") 
fig.show()

# %% [markdown]
# ## 2. Train, test and evaluate model for building magnetic data

# %%
# Getting the building mag feature view
version=1
feature_view_mag_building = fs.get_feature_view("hist_building_mag_fv", version=version)

# %%
# Get the latest version of the training dataset
t_X_train, t_X_test, t_y_train, t_y_test = feature_view_mag_building.train_test_split(0.2)

# %%
# Check the distribution of the target variable
t_y_train.value_counts(normalize=True)

# %%
# Define the model
t_model = KNeighborsClassifier(n_neighbors=2)
t_model.fit(t_X_train, t_y_train.values.ravel())

# %%
# Make predictions on the test set
t_y_pred = t_model.predict(t_X_test)
t_y_pred

# %%
# Make classification report
t_metrics = classification_report(t_y_test, t_y_pred, output_dict=True)
t_metrics

# %%
# Save the results in the model schema
t_results = confusion_matrix(t_y_test, t_y_pred)
t_results

# %%
# Make and save the confusion matrix
t_df_cm = pd.DataFrame(t_results, ['True Detection', 'True no Detection'],
                     ['Pred Detection', 'Pred no Detection'])

t_cm = sns.heatmap(t_df_cm, annot=True, fmt='g')

fig = t_cm.get_figure()
fig.savefig("/workspaces/2nd_semester_project/pictures/knn_mag_building_confusion_matrix.png") 
fig.show()

# %% [markdown]
# # Doing the same for radar data

# %%
# Get the latest version of the radar data feature view for bikelane
version=1
feature_view_rad_bikelane = fs.get_feature_view("hist_bikelane_radar_fv", version=version)
# Get the latest version of the training dataset
rad_bike_X_train, rad_bike_X_test, rad_bike_y_train, rad_bike_y_test = feature_view_rad_bikelane.train_test_split(0.2)

# %%
# Check the distribution of the target variable
rad_bike_y_train.value_counts(normalize=True)

# %%
# Define the model
rad_bike_model = KNeighborsClassifier(n_neighbors=2)
rad_bike_model.fit(rad_bike_X_train, rad_bike_y_train.values.ravel())

# %%
# Make predictions on the test set
rad_bike_y_pred = rad_bike_model.predict(rad_bike_X_test)
rad_bike_y_pred

# %%
# Make classification report
rad_bike_metrics = classification_report(rad_bike_y_test, rad_bike_y_pred, output_dict=True)
rad_bike_metrics

# %%
# Save the results in the model schema
rad_bike_results = confusion_matrix(rad_bike_y_test, rad_bike_y_pred)
rad_bike_results

# %%
# Make and save the confusion matrix
rad_bike_df_cm = pd.DataFrame(rad_bike_results, ['True Detection', 'True no Detection'],
                     ['Pred Detection', 'Pred no Detection'])

rad_bike_cm = sns.heatmap(rad_bike_df_cm, annot=True, fmt='g')

fig = rad_bike_cm.get_figure()
fig.savefig("/workspaces/2nd_semester_project/pictures/knn_rad_bike_confusion_matrix.png") 
fig.show()

# %% [markdown]
# ### Building radar data

# %%
# Get the latest version of the radar data feature view for building
version=1
feature_view_rad_building = fs.get_feature_view("hist_building_radar_fv", version=version)
# Get the latest version of the training dataset
rad_building_X_train, rad_building_X_test, rad_building_y_train, rad_building_y_test = feature_view_rad_building.train_test_split(0.2)

# %%
# Check the distribution of the target variable
rad_building_y_train.value_counts(normalize=True)

# %%
# Define the model
rad_building_model = KNeighborsClassifier(n_neighbors=2)
rad_building_model.fit(rad_building_X_train, rad_building_y_train.values.ravel())

# %%
# Make predictions on the test set
rad_building_y_pred = rad_building_model.predict(rad_building_X_test)
rad_building_y_pred

# %%
# Make classification report
rad_building_metrics = classification_report(rad_building_y_test, rad_building_y_pred, output_dict=True)
rad_building_metrics

# %%
# Save the results in the model schema
rad_building_results = confusion_matrix(rad_building_y_test, rad_building_y_pred)
rad_building_results

# %%
# Make and save the confusion matrix
rad_building_df_cm = pd.DataFrame(rad_building_results, ['True Detection', 'True no Detection'],
                     ['Pred Detection', 'Pred no Detection'])

rad_building_cm = sns.heatmap(rad_building_df_cm, annot=True, fmt='g')

fig = rad_building_cm.get_figure()
fig.savefig("/workspaces/2nd_semester_project/pictures/knn_rad_building_confusion_matrix.png")
fig.show()

# %% [markdown]
# ## 4. Uploading/updating models in the Feature Store

# %%
mr = project.get_model_registry()

# %%
# uploading the building_mag_hist_model to the model registry
model_dir="models"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(model, model_dir + "/building_mag_hist_model.pkl")
shutil.copyfile("/workspaces/2nd_semester_project/pictures/knn_mag_building_confusion_matrix.png", model_dir + "/knn_mag_building_confusion_matrix.png")

input_example = X_train.sample()
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

building_mag_hist_model = mr.python.create_model(
    version=2,
    name="building_mag_hist_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the building with magnetic data",)

building_mag_hist_model.save(model_dir)

# %%
# uploading the bikelane_mag_hist_model to the model registry
model_dir="models"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(t_model, model_dir + "/bikelane_mag_hist_model.pkl")
shutil.copyfile("/workspaces/2nd_semester_project/pictures/knn_mag_bikelane_matrix.png", model_dir + "/knn_mag_bikelane_matrix.png")

input_example = t_X_train.sample()
input_schema = Schema(t_X_train)
output_schema = Schema(t_y_train)
model_schema = ModelSchema(input_schema, output_schema)

bikelane_mag_hist_model = mr.python.create_model(
    version=2,
    name="bikelane_mag_hist_model", 
    metrics={"accuracy" : t_metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the bikelane with magnetic data",)

bikelane_mag_hist_model.save(model_dir)



# %%
# uploading the building_rad_hist_model to the model registry
model_dir="models"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(rad_building_model, model_dir + "/building_rad_hist_model.pkl")
shutil.copyfile("/workspaces/2nd_semester_project/pictures/knn_rad_building_confusion_matrix.png", model_dir + "/knn_rad_building_confusion_matrix.png")

input_example = rad_building_X_train.sample()
input_schema = Schema(rad_building_X_train)
output_schema = Schema(rad_building_y_train)
model_schema = ModelSchema(input_schema, output_schema)

building_rad_hist_model = mr.python.create_model(
    version=2,
    name="building_rad_hist_model", 
    metrics={"accuracy" : rad_building_metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the building with radar data",)

building_rad_hist_model.save(model_dir)


# %%
# uploading the bikelane_rad_hist_model to the model registry
model_dir="models"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(rad_bike_model, model_dir + "/bikelane_rad_hist_model.pkl")
shutil.copyfile("/workspaces/2nd_semester_project/pictures/knn_rad_bike_confusion_matrix.png", model_dir + "/knn_rad_bike_confusion_matrix.png")

input_example = rad_bike_X_train.sample()
input_schema = Schema(rad_bike_X_train)
output_schema = Schema(rad_bike_y_train)
model_schema = ModelSchema(input_schema, output_schema)

bikelane_rad_hist_model = mr.python.create_model(
    version=2,
    name="bikelane_rad_hist_model", 
    metrics={"accuracy" : rad_bike_metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the bikelane with radar data",)

bikelane_rad_hist_model.save(model_dir)


# %% [markdown]
# ## **Next up:** 5: Inference pipeline
# Go to the 5_inference_pipeline.ipynb notebook


