import pandas as pd
import ast
import re

# Read the CSV file into a DataFrame (replace with your actual CSV filename)
df = pd.read_csv('1b) Discovered_(Mis)Matches.csv')


# General function to clean and parse the string representation of the dictionary
def clean_and_parse_attribute(attribute, keys, prefix):
    if not isinstance(attribute, str):
        return pd.Series({f'{prefix}_{key}': None for key in keys})

    # Remove the 'u' prefix in Unicode strings (if it exists)
    attribute = re.sub(r"u'([a-zA-Z0-9_]+)'", r"'\1'", attribute)

    try:
        parsed_dict = ast.literal_eval(attribute)
    except Exception as e:
        print(f"Error parsing attribute: {attribute}. Error: {e}")
        return pd.Series({f'{prefix}_{key}': None for key in keys})

    return pd.Series({f'{prefix}_{key}': parsed_dict.get(key, None) for key in keys})


# General function to clean and parse the list of categories in 'foursquare_categories'
def clean_and_parse_foursquare_categories(attribute):
    if not isinstance(attribute, str):
        return pd.Series()

    # Remove the 'u' prefix in Unicode strings (if it exists)
    attribute = re.sub(r"u'([a-zA-Z0-9_]+)'", r"'\1'", attribute)

    try:
        category_list = ast.literal_eval(attribute)
    except Exception as e:
        print(f"Error parsing attribute: {attribute}. Error: {e}")
        return pd.Series()

    # Extract relevant details from each category
    categories_data = []
    for idx, category in enumerate(category_list):
        category_data = {
            f'foursquare_category_{idx + 1}_id': category.get('id', None),
            f'foursquare_category_{idx + 1}_name': category.get('name', None),
            f'foursquare_category_{idx + 1}_short_name': category.get('short_name', None),
            f'foursquare_category_{idx + 1}_plural_name': category.get('plural_name', None),
            f'foursquare_category_{idx + 1}_icon_prefix': category.get('icon', {}).get('prefix', None),
            f'foursquare_category_{idx + 1}_icon_suffix': category.get('icon', {}).get('suffix', None)
        }
        categories_data.append(category_data)

    # If no categories, return a series of NaNs
    if not categories_data:
        return pd.Series()

    # Combine all the category columns into a single row
    return pd.Series({key: value for data in categories_data for key, value in data.items()})


# General function to clean and parse the 'foursquare_social_media' dictionary
def clean_and_parse_social_media(attribute):
    if not isinstance(attribute, str):
        return pd.Series({'foursquare_social_media_facebook_id': None, 'foursquare_social_media_twitter': None})

    # Remove the 'u' prefix in Unicode strings (if it exists)
    attribute = re.sub(r"u'([a-zA-Z0-9_]+)'", r"'\1'", attribute)

    try:
        social_media_dict = ast.literal_eval(attribute)
    except Exception as e:
        print(f"Error parsing attribute: {attribute}. Error: {e}")
        return pd.Series({'foursquare_social_media_facebook_id': None, 'foursquare_social_media_twitter': None})

    # Return the facebook_id and twitter, if available
    return pd.Series({
        'foursquare_social_media_facebook_id': social_media_dict.get('facebook_id', None),
        'foursquare_social_media_twitter': social_media_dict.get('twitter', None)
    })


# Define the keys for different attributes
business_parking_keys = ['valet', 'garage', 'street', 'lot', 'validated']
ambience_keys = ['romantic', 'intimate', 'touristy', 'hipster', 'divey', 'classy', 'trendy', 'upscale', 'casual']
music_keys = ['dj', 'background_music', 'no_music', 'jukebox', 'live', 'video', 'karaoke']
best_nights_keys = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
good_for_meal_keys = ['dessert', 'latenight', 'lunch', 'dinner', 'brunch', 'breakfast']

# Apply the function to the respective columns for the 'attributes' columns
df[[f'attributes.BusinessParking_{key}' for key in business_parking_keys]] = df['attributes.BusinessParking'].apply(
    clean_and_parse_attribute, args=(business_parking_keys, 'BusinessParking'))

df[[f'attributes.Ambience_{key}' for key in ambience_keys]] = df['attributes.Ambience'].apply(
    clean_and_parse_attribute, args=(ambience_keys, 'Ambience'))

df[[f'attributes.Music_{key}' for key in music_keys]] = df['attributes.Music'].apply(
    clean_and_parse_attribute, args=(music_keys, 'Music'))

df[[f'attributes.BestNights_{key}' for key in best_nights_keys]] = df['attributes.BestNights'].apply(
    clean_and_parse_attribute, args=(best_nights_keys, 'BestNights'))

df[[f'attributes.GoodForMeal_{key}' for key in good_for_meal_keys]] = df['attributes.GoodForMeal'].apply(
    clean_and_parse_attribute, args=(good_for_meal_keys, 'GoodForMeal'))

# Apply the function to the 'foursquare_categories' column to split it into multiple columns
df = df.join(df['foursquare_categories'].apply(clean_and_parse_foursquare_categories))

# Apply the function to the 'foursquare_social_media' column to split it into separate columns
df = df.join(df['foursquare_social_media'].apply(clean_and_parse_social_media))

# Drop the original columns if no longer needed
df.drop(columns=['attributes.BusinessParking', 'attributes.Ambience', 'attributes.Music', 'attributes.BestNights',
                 'attributes.GoodForMeal', 'foursquare_categories', 'foursquare_social_media'], inplace=True)



def clean_and_parse_foursquare_features(attribute):
    if not isinstance(attribute, str):
        return pd.Series()

    # Remove the 'u' prefix in Unicode strings (if it exists)
    attribute = re.sub(r"u'([a-zA-Z0-9_]+)'", r"'\1'", attribute)

    try:
        features_dict = ast.literal_eval(attribute)
    except Exception as e:
        print(f"Error parsing attribute: {attribute}. Error: {e}")
        return pd.Series()

    features_data = {}

    # Extract payment details
    if 'payment' in features_dict:
        payment = features_dict['payment']
        if 'credit_cards' in payment:
            credit_cards = payment['credit_cards']
            for key in credit_cards:
                features_data[f'payment_credit_cards_{key}'] = credit_cards.get(key, None)
        if 'digital_wallet' in payment:
            digital_wallet = payment['digital_wallet']
            for key in digital_wallet:
                features_data[f'payment_digital_wallet_{key}'] = digital_wallet.get(key, None)

    # Extract services details
    if 'services' in features_dict:
        services = features_dict['services']
        features_data['services_dine_in'] = services.get('dine_in', None)  # Store dine_in separately
        for key in services:
            if key != 'dine_in':  # Exclude dine_in for separate handling
                features_data[f'services_{key}'] = services.get(key, None)

    # Extract amenities details
    if 'amenities' in features_dict:
        amenities = features_dict['amenities']
        features_data['amenities_parking'] = amenities.get('parking', None)  # Store parking separately
        for key in amenities:
            if key != 'parking':  # Exclude parking for separate handling
                features_data[f'amenities_{key}'] = amenities.get(key, None)

    # Extract food_and_drink details
    if 'food_and_drink' in features_dict:
        food_and_drink = features_dict['food_and_drink']
        if 'meals' in food_and_drink:
            meals = food_and_drink['meals']
            for key in meals:
                features_data[f'food_and_drink_meals_{key}'] = meals.get(key, None)
        if 'alcohol' in food_and_drink:
            alcohol = food_and_drink['alcohol']
            for key in alcohol:
                features_data[f'food_and_drink_alcohol_{key}'] = alcohol.get(key, None)

    # Extract attributes details
    if 'attributes' in features_dict:
        attributes = features_dict['attributes']
        for key in attributes:
            features_data[f'attributes_{key}'] = attributes.get(key, None)

    return pd.Series(features_data)

# Parse 'services_dine_in' column
def clean_and_parse_services_dine_in(attribute):
    if isinstance(attribute, dict):  # If already a dictionary, no need to parse
        dine_in_dict = attribute
    elif isinstance(attribute, str):
        try:
            dine_in_dict = ast.literal_eval(attribute)
        except Exception as e:
            print(f"Error parsing attribute: {attribute}. Error: {e}")
            return pd.Series()
    else:
        return pd.Series()

    dine_in_data = {
        'services_dine_in_reservations': dine_in_dict.get('reservations', None),
        'services_dine_in_groups_only_reservations': dine_in_dict.get('groups_only_reservations', None),
        'services_dine_in_essential_reservations': dine_in_dict.get('essential_reservations', None),
        'services_dine_in_online_reservations': dine_in_dict.get('online_reservations', None)
    }

    return pd.Series(dine_in_data)

# Parse 'amenities_parking' column
def clean_and_parse_amenities_parking(attribute):
    if isinstance(attribute, dict):  # If already a dictionary, no need to parse
        parking_dict = attribute
    elif isinstance(attribute, str):
        try:
            parking_dict = ast.literal_eval(attribute)
        except Exception as e:
            print(f"Error parsing attribute: {attribute}. Error: {e}")
            return pd.Series()
    else:
        return pd.Series()

    parking_data = {
        'amenities_parking_parking': parking_dict.get('parking', None),
        'amenities_parking_public_lot': parking_dict.get('public_lot', None),
        'amenities_parking_private_lot': parking_dict.get('private_lot', None),
        'amenities_parking_street_parking': parking_dict.get('street_parking', None)
    }

    return pd.Series(parking_data)

# Step 1: Parse 'foursquare_features'
df = df.join(df['foursquare_features'].apply(clean_and_parse_foursquare_features))
df.drop(columns=['foursquare_features'], inplace=True, errors='ignore')

# Debug: Check resulting columns
print(df[['services_dine_in', 'amenities_parking']].head())

# Step 2: Parse 'services_dine_in' column (from foursquare_features)
if 'services_dine_in' in df.columns:
    df = df.join(df['services_dine_in'].apply(clean_and_parse_services_dine_in))
    df.drop(columns=['services_dine_in'], inplace=True, errors='ignore')

# Step 3: Parse 'amenities_parking' column (from foursquare_features)
if 'amenities_parking' in df.columns:
    df = df.join(df['amenities_parking'].apply(clean_and_parse_amenities_parking))
    df.drop(columns=['amenities_parking'], inplace=True, errors='ignore')



# Save the updated DataFrame back to CSV (replace 'output_file_with_all_attributes.csv' with your desired output file name)
df.to_csv('1b) Discovered_(Mis)Matches.csv', index=False)

# Display the DataFrame (optional)
# print(df)
