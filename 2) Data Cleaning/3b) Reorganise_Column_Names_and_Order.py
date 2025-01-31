import pandas as pd

# Load your dataset
df = pd.read_csv('2b) Resstructured_JSON_from_Merged_Data.csv')

# Define the new column order and renaming mapping
column_mapping = {
    # Identifiers and Metadata
    'business_id': 'yelp_business_id',
    'fsq_id': 'fsq_id',
    'foursquare_store_id': 'fsq_store_id',
    'name': 'yelp_name',
    'foursquare_name': 'fsq_name',
    'address': 'yelp_address',
    'foursquare_address': 'fsq_address',
    'city': 'yelp_city',
    'state': 'yelp_state',
    'postal_code': 'yelp_postal_code',
    'latitude': 'yelp_lat',
    'longitude': 'yelp_lng',
    'foursquare_latitude': 'fsq_lat',
    'foursquare_longitude': 'fsq_lng',

    # Business Information
    'categories': 'yelp_categories',
    'is_open': 'yelp_is_open',

    # Important Statistics
    'stars': 'yelp_stars',
    'attributes.RestaurantsPriceRange2': 'yelp_price',
    'foursquare_price': 'fsq_price',
    'review_count': 'yelp_review_count',
    'foursquare_rating': 'fsq_rating',
    'foursquare_popularity': 'fsq_popularity',

    # Quality
    'attributes_service_quality': 'fsq_service_quality',
    'attributes_value_for_money': 'fsq_value_for_money',

    # Hours of Operation
    'hours': 'yelp_hours',
    'hours.Monday': 'yelp_hours_monday',
    'hours.Tuesday': 'yelp_hours_tuesday',
    'hours.Wednesday': 'yelp_hours_wednesday',
    'hours.Thursday': 'yelp_hours_thursday',
    'hours.Friday': 'yelp_hours_friday',
    'hours.Saturday': 'yelp_hours_saturday',
    'hours.Sunday': 'yelp_hours_sunday',

    # Best Nights
    'attributes.BestNights_monday': 'yelp_best_night_monday',
    'attributes.BestNights_tuesday': 'yelp_best_night_tuesday',
    'attributes.BestNights_wednesday': 'yelp_best_night_wednesday',
    'attributes.BestNights_thursday': 'yelp_best_night_thursday',
    'attributes.BestNights_friday': 'yelp_best_night_friday',
    'attributes.BestNights_saturday': 'yelp_best_night_saturday',
    'attributes.BestNights_sunday': 'yelp_best_night_sunday',

    # Parking
    'attributes.BikeParking': 'yelp_bike_parking',
    'attributes.BusinessParking_valet': 'yelp_parking_valet',
    'attributes.BusinessParking_garage': 'yelp_parking_garage',
    'attributes.BusinessParking_street': 'yelp_parking_street',
    'attributes.BusinessParking_lot': 'yelp_parking_lot',
    'attributes.BusinessParking_validated': 'yelp_parking_validated',
    'amenities_parking_parking': 'fsq_parking',
    'amenities_parking_public_lot': 'fsq_parking_public_lot',
    'amenities_parking_private_lot': 'fsq_parking_private_lot',
    'amenities_parking_street_parking': 'fsq_parking_street',

    # Ambiance
    'attributes.Ambience_romantic': 'yelp_ambiance_romantic',
    'attributes.Ambience_intimate': 'yelp_ambiance_intimate',
    'attributes.Ambience_touristy': 'yelp_ambiance_touristy',
    'attributes.Ambience_hipster': 'yelp_ambiance_hipster',
    'attributes.Ambience_divey': 'yelp_ambiance_divey',
    'attributes.Ambience_classy': 'yelp_ambiance_classy',
    'attributes.Ambience_trendy': 'yelp_ambiance_trendy',
    'attributes.Ambience_upscale': 'yelp_ambiance_upscale',
    'attributes.Ambience_casual': 'yelp_ambiance_casual',
    'attributes_trendy': 'fsq_ambiance_trendy',
    'attributes_romantic': 'fsq_ambiance_romantic',
    'attributes_late_night': 'fsq_late_night',

    # Music
    'attributes.Music_dj': 'yelp_music_dj',
    'attributes.Music_background_music': 'yelp_music_background',
    'attributes.Music_no_music': 'yelp_no_music',
    'amenities_music': 'fsq_music',
    'attributes.Music_jukebox': 'yelp_music_jukebox',
    'amenities_jukebox': 'fsq_music_jukebox',
    'attributes.Music_live': 'yelp_music_live',
    'amenities_live_music': 'fsq_music_live',
    'attributes.Music_video': 'yelp_music_video', #?????????
    'attributes.Music_karaoke': 'yelp_music_karaoke',

    # Good for Meals
    'attributes.GoodForMeal_breakfast': 'yelp_meals_breakfast',
    'attributes.GoodForMeal_brunch': 'yelp_meals_brunch',
    'attributes.GoodForMeal_lunch': 'yelp_meals_lunch',
    'attributes.GoodForMeal_dinner': 'yelp_meals_dinner',
    'attributes.GoodForMeal_dessert': 'yelp_meals_dessert',
    'attributes.GoodForMeal_latenight': 'yelp_meals_latenight',
    'food_and_drink_meals_breakfast': 'fsq_meals_breakfast',
    'food_and_drink_meals_brunch': 'fsq_meals_brunch',
    'food_and_drink_meals_lunch': 'fsq_meals_lunch',
    'food_and_drink_meals_dinner': 'fsq_meals_dinner',
    'food_and_drink_meals_dessert': 'fsq_meals_dessert',
    'attributes_quick_bite': 'fsq_meals_quick_bite',

    # Visit Information
    'attributes.RestaurantsTakeOut': 'yelp_takeout',
    'services_takeout': 'fsq_takeout',
    'attributes.RestaurantsReservations': 'yelp_reservations',
    'services_dine_in_reservations': 'fsq_reservations',
    'services_dine_in_online_reservations': 'fsq_online_reservations',
    'services_dine_in_groups_only_reservations': 'fsq_groups_only_reservations',
    'attributes.ByAppointmentOnly': 'yelp_appointments_only',
    'attributes.RestaurantsDelivery': 'yelp_delivery',
    'services_delivery': 'fsq_delivery',
    'attributes.DriveThru': 'yelp_drive_thru',
    'services_drive_through': 'fsq_drive_thru',

    # Providance
    'amenities_private_room': 'fsq_private_room',
    'amenities_sit_down_dining': 'fsq_sit_down_dining',
    'attributes.OutdoorSeating': 'yelp_outdoor_seating',
    'amenities_outdoor_seating': 'fsq_outdoor_seating',
    'attributes.WiFi': 'yelp_wifi',
    'amenities_wifi': 'fsq_wifi',
    'attributes.WheelchairAccessible': 'yelp_wheelchair_accessible',
    'amenities_wheelchair_accessible': 'fsq_wheelchair_accessible',
    'attributes.HasTV': 'yelp_tv',
    'amenities_tvs': 'fsq_tv',
    'amenities_restroom': 'fsq_restroom',

    # Service
    'attributes.RestaurantsTableService': 'yelp_table_service',
    'attributes.Caters': 'yelp_caters',

    # Attributes
    'attributes.NoiseLevel': 'yelp_noise_level',
    'attributes_noisy': 'fsq_noise_level',
    'attributes_clean': 'fsq_clean',

    # Target Groups
    'attributes.GoodForKids': 'yelp_good_for_kids',
    'attributes.RestaurantsGoodForGroups': 'yelp_good_for_groups',
    'attributes_groups_popular': 'fsq_groups_popular',
    'attributes_singles_popular': 'fsq_singles_popular',
    'attributes_crowded': 'fsq_crowded',

    # Clothing/Attire
    'attributes.RestaurantsAttire': 'yelp_attire',
    'attributes.CoatCheck': 'yelp_coat_check',
    'amenities_coat_check': 'fsq_coat_check',
    'attributes_dressy': 'fsq_attire_dressy',

    # Occasions
    'attributes_business_meeting': 'fsq_business_meeting',
    'attributes_dates_popular': 'fsq_dates_popular',
    'attributes_families_popular': 'fsq_families_popular',
    'attributes_special_occasion': 'fsq_special_occasion',

    # Allowances
    'attributes.DogsAllowed': 'yelp_dogs',
    'attributes_good_for_dogs': 'fsq_dogs',
    'attributes.Smoking': 'yelp_smoking',
    'amenities_smoking': 'fsq_smoking',
    'attributes.AgesAllowed': 'yelp_age',

    # Drinking
    'attributes.HappyHour': 'yelp_happy_hour',
    'attributes.Alcohol': 'yelp_alcohol',
    'food_and_drink_meals_happy_hour': 'fsq_happy_hour',
    'food_and_drink_alcohol_cocktails': 'fsq_cocktails',
    'food_and_drink_alcohol_full_bar': 'fsq_full_bar',
    'food_and_drink_alcohol_beer': 'fsq_beer',
    'food_and_drink_alcohol_wine': 'fsq_wine',
    'food_and_drink_alcohol_bar_service': 'fsq_bar_service',

    # Dietary Preferences
    'attributes.DietaryRestrictions': 'yelp_dietary_restrictions',
    'attributes_healthy_diet': 'fsq_healthy',
    'attributes_vegan_diet': 'fsq_vegan_diet',
    'attributes_gluten_free_diet': 'fsq_gluten_free_diet',
    'attributes_vegetarian_diet': 'fsq_vegetarian_diet',

    # Payment Options
    'attributes.BusinessAcceptsCreditCards': 'yelp_credit_cards',
    'attributes.BusinessAcceptsBitcoin': 'yelp_accepts_bitcoin',
    'payment_credit_cards_accepts_credit_cards': 'fsq_credit_cards',
    'payment_digital_wallet_accepts_nfc': 'fsq_nfc',
    'payment_credit_cards_amex': 'fsq_credit_cards_amex',
    'payment_credit_cards_discover': 'fsq_credit_cards_discover',
    'payment_credit_cards_visa': 'fsq_credit_cards_visa',
    'payment_credit_cards_diners_club': 'fsq_credit_cards_diners_club',
    'payment_credit_cards_master_card': 'fsq_credit_cards_master_card',
    'payment_credit_cards_union_pay': 'fsq_credit_cards_union_pay',

    # Social Media
    'foursquare_social_media_facebook_id': 'fsq_facebook',
    'foursquare_social_media_twitter': 'fsq_twitter',

    # Generated Columns
    'clean_name': 'generated_clean_name',
    'clean_foursquare_name': 'generated_clean_foursquare_name',
    'similarity_score': 'generated_similarity_score',
    'match_category': 'generated_match_category'
}

# Apply the renaming and reordering
def reorganize_columns(df, column_mapping):
    # Rename columns
    df = df.rename(columns=column_mapping)

    # Reorder columns
    ordered_columns = list(column_mapping.values())
    df = df[ordered_columns]

    return df

# Example Usage
df = reorganize_columns(df, column_mapping)

# Save the updated dataset
df.to_csv('3c) Reorganised_Column_Names_and_Order.csv', index=False)
