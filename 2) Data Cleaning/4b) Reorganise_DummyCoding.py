import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file
file_path = "3c) Reorganised_Column_Names_and_Order.csv"
df = pd.read_csv(file_path)

# Define the mapping for 'Average', 'Great', 'Poor' (No after comma numbers)
stage_mapping = {'Great': 2, 'Average': 1, 'Poor': 0}
stage_columns = [
    'fsq_service_quality', 'fsq_value_for_money', 'fsq_ambiance_trendy',
    'fsq_ambiance_romantic', 'fsq_late_night', 'fsq_meals_quick_bite',
    'yelp_noise_level', 'fsq_noise_level', 'fsq_clean', 'fsq_groups_popular',
    'fsq_singles_popular', 'fsq_crowded', 'fsq_attire_dressy',
    'fsq_business_meeting', 'fsq_dates_popular', 'fsq_families_popular',
    'fsq_special_occasion', 'fsq_dogs', 'fsq_healthy', 'fsq_vegan_diet',
    'fsq_gluten_free_diet', 'fsq_vegetarian_diet'
]
# Apply the mapping to these columns
for col in stage_columns:
    if col in df.columns:
        df[col] = df[col].map(stage_mapping).astype('Int64')  # Convert to integer without decimals

# Dummy coding for TRUE/FALSE columns (No after comma numbers)
bool_columns = [
    'yelp_best_night_monday', 'yelp_best_night_tuesday', 'yelp_best_night_wednesday',
    'yelp_best_night_thursday', 'yelp_best_night_friday', 'yelp_best_night_saturday',
    'yelp_bike_parking', 'yelp_parking_valet', 'yelp_parking_garage',
    'yelp_parking_street', 'yelp_parking_lot', 'yelp_parking_validated',
    'fsq_parking', 'fsq_parking_public_lot', 'fsq_parking_private_lot',
    'fsq_parking_street', 'yelp_ambiance_romantic', 'yelp_ambiance_intimate',
    'yelp_ambiance_touristy', 'yelp_ambiance_hipster', 'yelp_ambiance_divey',
    'yelp_ambiance_classy', 'yelp_ambiance_trendy', 'yelp_ambiance_upscale',
    'yelp_ambiance_casual', 'fsq_music', 'yelp_music_dj', 'yelp_music_background',
    'yelp_no_music', 'yelp_music_jukebox', 'fsq_music_jukebox', 'yelp_music_live',
    'fsq_music_live', 'yelp_music_video', 'yelp_music_karaoke', 'yelp_meals_breakfast',
    'yelp_meals_brunch', 'yelp_meals_lunch', 'yelp_meals_dinner', 'yelp_meals_dessert',
    'yelp_meals_latenight', 'fsq_takeout', 'yelp_takeout', 'yelp_reservations',
    'fsq_reservations', 'fsq_online_reservations', 'fsq_groups_only_reservations',
    'yelp_delivery', 'fsq_delivery', 'yelp_drive_thru', 'fsq_drive_thru',
    'fsq_private_room', 'fsq_sit_down_dining', 'yelp_outdoor_seating',
    'fsq_outdoor_seating', 'fsq_wheelchair_accessible', 'yelp_wheelchair_accessible',
    'yelp_tv', 'fsq_tv', 'fsq_restroom', 'yelp_table_service', 'yelp_caters',
    'yelp_good_for_kids', 'yelp_good_for_groups', 'yelp_coat_check', 'fsq_coat_check',
    'yelp_dogs', 'fsq_happy_hour', 'fsq_cocktails', 'fsq_full_bar', 'fsq_beer',
    'fsq_wine', 'fsq_bar_service', 'yelp_credit_cards', 'yelp_accepts_bitcoin',
    'fsq_credit_cards', 'fsq_nfc', 'fsq_credit_cards_amex', 'fsq_credit_cards_discover',
    'fsq_credit_cards_visa', 'fsq_credit_cards_diners_club', 'fsq_credit_cards_master_card',
    'fsq_credit_cards_union_pay', 'yelp_appointments_only', 'yelp_best_night_sunday',
    'fsq_meals_breakfast', 'fsq_meals_brunch', 'fsq_meals_lunch', 'fsq_meals_dinner',
    'fsq_meals_dessert', 'yelp_happy_hour'
]
# Apply dummy coding for TRUE/FALSE columns and ensure it returns `1` or `0`
for col in bool_columns:
    if col in df.columns:
        # Convert TRUE/FALSE values to 1 or 0 (as integers)
        df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['true', '1', 'yes', True] else (0 if pd.notna(x) else None)).astype('Int64')  # Ensure integer output






# Function to clean up and standardize values (for some reason this did not work)
# Create a mapping for each column's categories
wifi_mapping = {
    'f': 'free', 'fp': 'free', 'n': 'no', 'p': 'paid', 't': 'paid'
}

attire_mapping = {
    'causal': 'casual', 'dressy': 'dressy', 'formal': 'formal'
}

smoking_mapping = {
    'no': 'no', 'outdoor': 'outdoor', 'yes': 'yes'
}

alcohol_mapping = {
    'beer_and_wine': 'beer_and_wine', 'full_bar': 'full_bar', 'none': 'none'
}


# Function to normalize values with mappings where applicable
def normalize_value(value, column_mapping=None):
    if pd.isna(value):
        return None  # Return None if value is NaN
    # Convert value to string, remove any leading 'u' if present, strip spaces, remove apostrophes, and make lowercase
    value = str(value).strip().lower()

    # Remove the 'u' prefix if it's there
    if value.startswith('u'):
        value = value[1:].strip()

    # Remove apostrophes
    value = value.replace("'", "")  # Remove apostrophes from the string

    # Apply the column-specific mapping if available
    if column_mapping and value in column_mapping:
        return column_mapping[value]

    return value  # Return the normalized value or original value


# Apply normalization to the columns with their respective mappings
df['yelp_wifi'] = df['yelp_wifi'].apply(normalize_value, column_mapping=wifi_mapping)
df['fsq_wifi'] = df['fsq_wifi'].apply(normalize_value, column_mapping=wifi_mapping)
df['yelp_attire'] = df['yelp_attire'].apply(normalize_value, column_mapping=attire_mapping)
df['yelp_smoking'] = df['yelp_smoking'].apply(normalize_value, column_mapping=smoking_mapping)
df['yelp_alcohol'] = df['yelp_alcohol'].apply(normalize_value, column_mapping=alcohol_mapping)

# Check the first few rows to verify
print(df[['yelp_wifi', 'fsq_wifi', 'yelp_attire', 'yelp_smoking', 'yelp_alcohol']].head())








# Create binary columns for Facebook and Twitter (No after comma numbers)
df['fsq_facebook'] = df['fsq_facebook'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else None)
df['fsq_twitter'] = df['fsq_twitter'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else None)

# Create the social_media column
def classify_social_media(row):
    if row['fsq_facebook'] == 1 and row['fsq_twitter'] == 1:
        return 2
    elif row['fsq_facebook'] == 1 or row['fsq_twitter'] == 1:
        return 1
    else:
        return None
df['generated_social_media'] = df.apply(classify_social_media, axis=1)

# Insert the 'social_media' column after the 'fsq_twitter' column
columns = list(df.columns)  # Get the current column order
twitter_index = columns.index('fsq_twitter')  # Find the index of 'fsq_twitter'
columns.insert(twitter_index + 1, columns.pop(columns.index('generated_social_media')))  # Insert 'social_media' after 'fsq_twitter'
df = df[columns]  # Reorder the DataFrame according to the new column order





# Function to calculate the total hours for a week
def calculate_total_hours(row):
    total_hours = timedelta(0)  # Initialize total hours as zero
    days = [
        'yelp_hours_monday',
        'yelp_hours_tuesday',
        'yelp_hours_wednesday',
        'yelp_hours_thursday',
        'yelp_hours_friday',
        'yelp_hours_saturday',
        'yelp_hours_sunday'
    ]

    for day in days:
        hours = str(row[day]).strip() if pd.notna(row[day]) else ""
        if hours and hours != "0:0-0:0":
            intervals = hours.split(",")  # Split by multiple intervals if any
            for interval in intervals:
                try:
                    start, end = interval.split("-")
                    start_time = datetime.strptime(start, "%H:%M")
                    end_time = datetime.strptime(end, "%H:%M")

                    # Adjust for intervals that cross midnight
                    if end_time < start_time:
                        end_time += timedelta(days=1)

                    total_hours += (end_time - start_time)
                except Exception as e:
                    print(f"Error processing interval '{interval}' in column '{day}': {e}")
    return total_hours.total_seconds() / 3600  # Convert total seconds to hours

# Apply the function to the dataset
df['yelp_hours'] = df.apply(calculate_total_hours, axis=1)

# Replace NaN with empty cells in the entire dataframe before saving to CSV
# Only replace NaN in columns where we specifically want it, such as for textual data
df = df.where(pd.notna(df), None)

# Save the updated dataset to a new CSV file
output_file = "4c) Reorganised_DummyCoding.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset with social media column saved to {output_file}")
