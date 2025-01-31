import pandas as pd
from sklearn.preprocessing import StandardScaler

# List of columns to remove
columns_to_remove = [
    "fsq_id", "fsq_store_id", "fsq_name", "yelp_address", "fsq_address", "fsq_lng", "fsq_lat",

    "yelp_categories",

    "yelp_hours_monday", "yelp_hours_tuesday", "yelp_hours_wednesday",
    "yelp_hours_thursday","yelp_hours_friday", "yelp_hours_saturday", "yelp_hours_sunday", "yelp_best_night_monday",
    "yelp_best_night_tuesday", "yelp_best_night_wednesday", "yelp_best_night_thursday",
    "yelp_best_night_friday", "yelp_best_night_saturday", "yelp_best_night_sunday",

    "yelp_bike_parking",

    "yelp_parking_valet", "yelp_parking_validated",

    "fsq_private_room",

    "fsq_sit_down_dining",

    "fsq_restroom",

    #"yelp_caters",

    "fsq_crowded",

    "fsq_attire_dressy",

    "fsq_business_meeting", "fsq_dates_popular", "fsq_families_popular", "fsq_special_occasion",

    "yelp_age",

    "fsq_cocktails", "fsq_full_bar", "fsq_beer", "fsq_wine", "fsq_bar_service",

    "yelp_dietary_restrictions", "fsq_healthy", "fsq_vegan_diet", "fsq_gluten_free_diet", "fsq_vegetarian_diet",

    "yelp_accepts_bitcoin", "fsq_nfc", "fsq_credit_cards_amex", "fsq_credit_cards_discover",
    "fsq_credit_cards_visa", "fsq_credit_cards_diners_club", "fsq_credit_cards_master_card",
    "fsq_credit_cards_union_pay",

    "generated_clean_name", "generated_clean_foursquare_name", "generated_similarity_score", "generated_match_category",

    "yelp_price", "fsq_price",

    "yelp_music_dj", "yelp_music_background", "yelp_no_music", "fsq_music", "yelp_music_jukebox",
    "fsq_music_jukebox", "yelp_music_live", "fsq_music_live", "yelp_music_video", "yelp_music_karaoke",

    "yelp_takeout", "fsq_takeout",

    "yelp_reservations", "fsq_reservations",

    "yelp_delivery", "fsq_delivery",

    "yelp_drive_thru", "fsq_drive_thru",

    "yelp_meals_breakfast", "fsq_meals_breakfast", "yelp_meals_brunch",
    "fsq_meals_brunch", "yelp_meals_lunch", "fsq_meals_lunch", "yelp_meals_dinner", "fsq_meals_dinner",
    "yelp_meals_dessert", "fsq_meals_dessert",

    "yelp_outdoor_seating", "fsq_outdoor_seating",

    "yelp_wifi", "fsq_wifi",

    "yelp_wheelchair_accessible", "fsq_wheelchair_accessible",

    "yelp_tv", "fsq_tv",

    "yelp_noise_level", "fsq_noise_level",

    "yelp_good_for_groups", "fsq_groups_popular",

    "yelp_dogs", "fsq_dogs",

    "yelp_smoking", "fsq_smoking",

    "yelp_happy_hour", "fsq_happy_hour",

    "yelp_credit_cards", "fsq_credit_cards",

    "yelp_parking_garage", "yelp_parking_street", "yelp_parking_lot", "fsq_parking",
    "fsq_parking_public_lot", "fsq_parking_private_lot", "fsq_parking_street",

    "yelp_ambiance_trendy", "fsq_ambiance_trendy", "yelp_ambiance_romantic",
    "fsq_ambiance_romantic",

    "fsq_online_reservations", "fsq_groups_only_reservations",

    "fsq_facebook", "fsq_twitter",

    "yelp_alcohol",

    "service_singles_popular",

    "yelp_appointments_only",

    "yelp_coat_check", "fsq_coat_check",

    "fsq_service_quality"
]

# File paths
input_file = "4c) Reorganised_DummyCoding.csv"
output_file = "5b) Fuse_and_Filter_Columns.csv"

# Load the CSV file
df = pd.read_csv(input_file)

# Filter rows where 'generated_match_category' is 'high match'
df = df[(df['generated_match_category'].isin(['high match', 'moderate match'])) & (df['yelp_is_open'] == 1)]
print(f"Number of rows: {df.shape[0]}")


# Fusion logic
# 1. Fuse yelp_price and fsq_price
def fuse_price(row):
    return row['yelp_price'] if pd.notna(row['yelp_price']) else row['fsq_price']
df['price'] = df.apply(fuse_price, axis=1)


# 3. Fuse binary columns
def fuse_binary(row, col1, col2):
    if pd.notna(row[col1]):
        return row[col1]
    else:
        return row[col2]

binary_column_pairs = [
    ("yelp_takeout", "fsq_takeout"),
    ("yelp_reservations", "fsq_reservations"),
    ("yelp_delivery", "fsq_delivery"),
    ("yelp_drive_thru", "fsq_drive_thru"),
    ("yelp_meals_breakfast", "fsq_meals_breakfast"),
    ("yelp_meals_brunch", "fsq_meals_brunch"),
    ("yelp_meals_lunch", "fsq_meals_lunch"),
    ("yelp_meals_dinner", "fsq_meals_dinner"),
    ("yelp_meals_dessert", "fsq_meals_dessert"),
    ("yelp_outdoor_seating", "fsq_outdoor_seating"),
    ("yelp_wifi", "fsq_wifi"),
    ("yelp_wheelchair_accessible", "fsq_wheelchair_accessible"),
    ("yelp_tv", "fsq_tv"),
    ("yelp_noise_level", "fsq_noise_level"),
    ("yelp_good_for_groups", "fsq_groups_popular"),
    ("yelp_dogs", "fsq_dogs"),
    ("yelp_smoking", "fsq_smoking"),
    ("yelp_happy_hour", "fsq_happy_hour"),
    ("yelp_credit_cards", "fsq_credit_cards"),
    ("yelp_ambiance_trendy", "fsq_ambiance_trendy"),
    ("yelp_ambiance_romantic", "fsq_ambiance_romantic")
]

for col1, col2 in binary_column_pairs:
    fused_col = col1.replace("yelp_", "")  # Derive new column name
    df[fused_col] = df.apply(lambda row: fuse_binary(row, col1, col2), axis=1)

# 4. Fuse parking columns
def fuse_parking(row):
    parking_columns = [
        "yelp_parking_garage", "yelp_parking_street", "yelp_parking_lot",
        "fsq_parking", "fsq_parking_public_lot", "fsq_parking_private_lot", "fsq_parking_street"
    ]
    if any(row[col] == 1 for col in parking_columns if pd.notna(row[col])):
        return 1
    elif any(row[col] == 0 for col in parking_columns if pd.notna(row[col])):
        return 0
    else:
        return None
df['parking'] = df.apply(fuse_parking, axis=1)

# 5. Adjust yelp_alcohol column
def transform_alcohol1(value):
    return 1 if pd.notna(value) and value.lower() != "none" else 0
df['alcohol'] = df['yelp_alcohol'].apply(transform_alcohol1)

df['alcohol'] = df.apply(
    lambda row: 1 if pd.isna(row['alcohol']) and row['happy_hour'] == 1
    else (0 if pd.isna(row['alcohol']) and row['happy_hour'] == 0 else row['alcohol']),
    axis=1
)




# Normalize the 'smoking' column
def normalize_smoking(value):
    if pd.isna(value):  # Handle NaN or None
        return None
    value = str(value).strip().lower()  # Ensure string type, strip spaces, and convert to lowercase
    if value == "TRUE":
        return "yes"
    elif value == "FALSE":
        return "no"
    elif value == "outdoor":
        return "outdoor"
    else:
        return None  # Default for unexpected values

# Apply the normalization to the column
if 'smoking' in df.columns:  # Ensure column exists
    df['smoking'] = df['smoking'].apply(normalize_smoking)
else:
    print("Column 'attr_smoking' does not exist in the DataFrame.")


# Drop the specified columns
df = df.drop(columns=columns_to_remove, errors='ignore')

# Remove yelp_ and fsq_ prefixes from remaining columns
df.columns = [col.replace("yelp_", "").replace("fsq_", "").replace("generated_", "") for col in df.columns]


# Define the prefixes and their associated columns
prefixes = {
    "business": ['business_id'],
    "week": ["hours"],
    "rating": ["stars", "review_count", "rating", "popularity", "price"],
    "social": ["social_media"],
    "ambiance": ["ambiance_intimate", "ambiance_touristy", "ambiance_hipster",
                 "ambiance_divey", "ambiance_classy", "ambiance_upscale",
                 "ambiance_casual", "ambiance_trendy", "ambiance_romantic"],
    "meal": ["meals_breakfast", "meals_brunch", "meals_lunch", "meals_dinner",
             "meals_dessert", "meals_latenight"],
    "attr": ["parking", "credit_cards", "outdoor_seating", "wifi", "tv"],
    "reservations": ["reservations"],
    "service": ["table_service", 'caters', "good_for_kids", "good_for_groups"],
    "collect": ["takeout", "delivery"],
    "alcohol": ["alcohol"]
}
# Generate renamed columns dynamically with a check for existing prefixes
renamed_columns = {}
for prefix, cols in prefixes.items():
    for col in cols:
        if not col.startswith(prefix):
            renamed_columns[col] = f"{prefix}_{col}"
        else:
            renamed_columns[col] = col

columns_to_keep = [f"{prefix}_{col}" if not col.startswith(prefix) else col
                   for prefix, cols in prefixes.items() for col in cols]
df.rename(columns=renamed_columns, inplace=True)
df = df[columns_to_keep]


threshold = 7
df['rating'] = (df['rating'] > threshold).astype(int)


# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["attr_wifi"])
# Create a single 'wifi' column: 1 if free or paid, 0 if no
df["wifi"] = df[["attr_wifi_free", "attr_wifi_paid"]].max(axis=1).astype(int)
# Drop the original one-hot encoded columns
df = df.drop(columns=['attr_wifi_free', 'attr_wifi_no', 'attr_wifi_paid'])

df = df[df['week_hours'] >= 5]

# Find all rows where "rating_popularity" > 1
greater_than_one = df[df["rating_popularity"] > 1]
print("Values greater than 1 in 'rating_popularity':")
print(greater_than_one["rating_popularity"])
df["rating_popularity"] = df["rating_popularity"].apply(
    lambda x: f"0.{int(x)}" if x > 1 else x)


df = df.T.drop_duplicates().T
df.to_csv(output_file, index=False)

print(f"Updated CSV saved to {output_file}")
