# Make sure you'll be running this script from the root of the project

# Install kaggle
pip install kaggle

# Create a folder for the data
mkdir data

# Download the data
kaggle datasets download -d saurabhshahane/road-traffic-accidents

# Unzip the data
unzip road-traffic-accidents.zip "RTA Dataset.csv" -d data

# Rename as raw_data.csv
mv data/"RTA Dataset.csv" data/raw_data.csv

# Remove the zip file
rm road-traffic-accidents.zip