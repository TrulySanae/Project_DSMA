import os
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from requests import Request, Session

# Load the .env file to access the API key
load_dotenv()
api_key = os.getenv("API_KEY_Foursquare")

# Define the API endpoint URL
url = "https://api.foursquare.com/v3/places/search"

# Function to get Foursquare data for a given restaurant
def get_foursquare_data(row):
    # Combine restaurant name and address for search
    query = f"{row['name']}"
    params = {
        "query": query,
        "ll": f"{row['latitude']},{row['longitude']}",  # Latitude and Longitude
        "limit": 1,  # Limit to 1 result for precision
        # Requesting the new list of fields
        "fields": "fsq_id,name,location,geocodes,categories,social_media,rating,popularity,price,menu,tastes,features,store_id"
    }
    headers = {
        "accept": "application/json",
        "Authorization": api_key
    }

    # Correctly format the URL with query parameters for debugging
    request = Request('GET', url, headers=headers, params=params)
    prepared = request.prepare()
    print(f"Request URL: {prepared.url}")  # Print the full URL

    try:
        # Send the request and process the response
        with Session() as session:
            response = session.send(prepared)
            response.raise_for_status()  # Check if the request was successful
            data = response.json().get("results", [])
            if data:
                place = data[0]
                return {
                    "fsq_id": place.get("fsq_id", ""),
                    "foursquare_name": place.get("name", ""),
                    "foursquare_address": place.get("location", {}).get("formatted_address", ""),
                    "foursquare_latitude": place.get("geocodes", {}).get("main", {}).get("latitude", ""),
                    "foursquare_longitude": place.get("geocodes", {}).get("main ", {}).get("longitude", ""),
                    "foursquare_categories": str(place.get("categories", [])),
                    "foursquare_social_media": str(place.get("social_media", {})),
                    "foursquare_rating": place.get("rating", ""),
                    "foursquare_popularity": place.get("popularity", ""),
                    "foursquare_price": place.get("price", ""),
                    "foursquare_menu": str(place.get("menu", {})),
                    "foursquare_tastes": str(place.get("tastes", [])),
                    "foursquare_features": str(place.get("features", [])),
                    "foursquare_store_id": place.get("store_id", "")
                }
        return None
    except requests.RequestException as e:
        print(f"Error for row {row['name']}: {e}")
        return None

# Read the CSV file into a DataFrame
df = pd.read_csv('4b) Distributed_Files/output_part_1.csv')

# Add new columns for Foursquare data
df["fsq_id"] = None
df["foursquare_name"] = None
df["foursquare_address"] = None
df["foursquare_latitude"] = None
df["foursquare_longitude"] = None
df["foursquare_categories"] = None
df["foursquare_social_media"] = None
df["foursquare_rating"] = None
df["foursquare_popularity"] = None
df["foursquare_price"] = None
df["foursquare_menu"] = None
df["foursquare_tastes"] = None
df["foursquare_features"] = None
df["foursquare_store_id"] = None

# Use ThreadPoolExecutor to fetch data in parallel for better performance
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for index, row in df.iterrows():
        future = executor.submit(get_foursquare_data, row)
        futures.append((index, future))

    # Add Foursquare data directly to the DataFrame as each future completes
    for index, future in futures:
        foursquare_data = future.result()
        if foursquare_data:
            # Update the DataFrame directly with the Foursquare data
            df.at[index, "fsq_id"] = foursquare_data["fsq_id"]
            df.at[index, "foursquare_name"] = foursquare_data["foursquare_name"]
            df.at[index, "foursquare_address"] = foursquare_data["foursquare_address"]
            df.at[index, "foursquare_latitude"] = foursquare_data["foursquare_latitude"]
            df.at[index, "foursquare_longitude"] = foursquare_data["foursquare_longitude"]
            df.at[index, "foursquare_categories"] = foursquare_data["foursquare_categories"]
            df.at[index, "foursquare_social_media"] = foursquare_data["foursquare_social_media"]
            df.at[index, "foursquare_rating"] = foursquare_data["foursquare_rating"]
            df.at[index, "foursquare_popularity"] = foursquare_data["foursquare_popularity"]
            df.at[index, "foursquare_price"] = foursquare_data["foursquare_price"]
            df.at[index, "foursquare_menu"] = foursquare_data["foursquare_menu"]
            df.at[index, "foursquare_tastes"] = foursquare_data["foursquare_tastes"]
            df.at[index, "foursquare_features"] = foursquare_data["foursquare_features"]
            df.at[index, "foursquare_store_id"] = foursquare_data["foursquare_store_id"]

# Save the updated DataFrame to a new CSV file
df.to_csv('6) Fully_Merged_Database_Foursquare_Data_222.csv', index=False)

print("CSV file updated successfully with Foursquare data.")
