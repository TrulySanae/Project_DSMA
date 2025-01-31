import pandas as pd
import json

# Load the data to examine its structure
file_path = '1) Full_JSON_Database_Data.csv'
data = pd.read_csv(file_path)

# Display a preview of the data
# data.head(), data.info()

# Parse the JSON strings into a structured DataFrame
data_parsed = data['j'].apply(json.loads)
structured_data = pd.json_normalize(data_parsed)

# Write the structured data to a new CSV file
output_file_path = 'Restructure_JSON_Database_Data.csv'
structured_data.to_csv(output_file_path, index=False)

# Display a preview of the structured data
structured_data.head(), structured_data.info()