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
project = hopsworks.login()
fs = project.get_feature_store()

# %% [markdown]
# ## 2. Train, test and evaluate model for building data

# %%
# Get the latest version of the feature view
version=1
feature_view = fs.get_feature_view("building_hist_fv", version=version)

# %%
# Get the latest version of the training dataset
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# %%
# Check the distribution of the target variable
y_train.value_counts(normalize=True)

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
fig.savefig("/workspaces/MLOps_Project/pictures/building_hist_confusion_matrix.png") 
fig.show()

# %% [markdown]
# ## 2. Train, test and evaluate model for bike lane data

# %%
# Getting the bikelane_hist_fv feature view
version=1
bikelane_feature_view = fs.get_feature_view("bikelane_hist_fv", version=version)

# %%
# Get the latest version of the training dataset
t_X_train, t_X_test, t_y_train, t_y_test = bikelane_feature_view.train_test_split(0.2)

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
fig.savefig("/workspaces/MLOps_Project/pictures/bikelane_hist_confusion_matrix.png") 
fig.show()

# %% [markdown]
# ## 4. Uploading/updating models in the Feature Store

# %%
mr = project.get_model_registry()

# %%

model_dir="models"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(model, model_dir + "/building_hist_model.pkl")
shutil.copyfile("/workspaces/MLOps_Project/pictures/building_hist_confusion_matrix.png", model_dir + "/building_hist_confusion_matrix.png")

input_example = X_train.sample()
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

building_hist_model = mr.python.create_model(
    version=2,
    name="building_hist_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the building",)

building_hist_model.save(model_dir)

# %%

if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(t_model, model_dir + "/bikelane_hist_model.pkl")
shutil.copyfile("/workspaces/MLOps_Project/pictures/bikelane_hist_confusion_matrix.png", model_dir + "/bikelane_hist_confusion_matrix.png")

input_example = t_X_train.sample()
input_schema = Schema(t_X_train)
output_schema = Schema(t_y_train)
model_schema = ModelSchema(input_schema, output_schema)

bikelane_hist_model = mr.python.create_model(
    version=1,
    name="bikelane_hist_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Predictions on the parking spot close to the bikelane",)

bikelane_hist_model.save(model_dir)

# %% [markdown]
# ## **Next up:** 5: Inference pipeline
# Go to the 5_inference_pipeline.ipynb notebook


