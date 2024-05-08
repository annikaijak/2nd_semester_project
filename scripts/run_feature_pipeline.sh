#!/bin/bash

set -e

cd notebooks/python_scripts
#set the API_PASSWORD
export API_PASSWORD=$API_PASSWORD

# Run the feature pipeline
python 2_latest_api_feature_pipeline.py

#jupyter nbconvert --to notebook --execute 2_feature_pipeline.ipynb
