import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer

# File paths
input_csv_path = "5b) Fuse_and_Filter_Columns.csv"
output_csv_path = "6b) Imputed_Dataset.csv"

# Step 1: Read the CSV file
df = pd.read_csv(input_csv_path)

# Ensure 'rating' and 'business_id' columns are excluded from imputation, but keep them in the DataFrame
rating_column = df['rating']  # Save the 'rating' column
business_id_column = df['business_id']  # Save the 'business_id' column
df = df.drop(columns=['rating', 'business_id'])  # Drop 'rating' and 'business_id' for imputation

# Step 2: Identify variable types
ordinal_columns = ['rating_price', 'social_media']
binary_columns = ['ambiance_intimate', 'ambiance_touristy', 'ambiance_hipster', 'ambiance_divey',
                  'ambiance_classy', 'ambiance_upscale', 'ambiance_casual', 'ambiance_trendy',
                  'ambiance_romantic', 'meals_breakfast', 'meals_brunch', 'meals_lunch', 'meals_dinner',
                  'meals_dessert', 'meals_latenight', 'attr_parking', 'attr_credit_cards',
                  'attr_outdoor_seating', 'attr_tv', 'reservations', 'service_table_service',
                  'service_caters', 'service_good_for_kids', 'service_good_for_groups', 'collect_takeout',
                  'collect_delivery', 'alcohol', 'wifi']
continuous_columns = ['week_hours', 'rating_stars', 'rating_review_count', 'rating_popularity']

# Step 3: Ensure the data is numeric
df[binary_columns] = df[binary_columns].astype(float)  # Binary as numeric (0 and 1)
df[ordinal_columns] = df[ordinal_columns].astype(float)  # Ordinal as numeric
df[continuous_columns] = df[continuous_columns].astype(float)  # Continuous as float

# Step 4: Initialize Iterative Imputer
imputer = IterativeImputer(max_iter=50, random_state=42)

# Step 5: Apply the imputer
df_imputed = imputer.fit_transform(df)

# Step 6: Convert the result back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Step 7: Re-include the 'rating' and 'business_id' columns (unchanged)
df_imputed['rating'] = rating_column  # Add 'rating' back without any changes
df_imputed['business_id'] = business_id_column  # Add 'business_id' back without any changes

# Step 8: Post-imputation adjustments
# Ensure binary variables are rounded to 0 or 1
for col in binary_columns:
    df_imputed[col] = df_imputed[col].round().clip(0, 1).astype(int)

# Ensure ordinal variables are rounded and clipped to their valid range
ordinal_ranges = {  # Define the valid range for each ordinal variable
    'rating_price': (1, 4),  # Example: ordinal_col1 values should be between 1 and 5
    'social_media': (0, 2)   # Example: ordinal_col2 values should be between 1 and 3
}
for col, (min_val, max_val) in ordinal_ranges.items():
    df_imputed[col] = df_imputed[col].round().clip(min_val, max_val).astype(int)

# Step 9: Save the imputed DataFrame to a CSV file
df_imputed.to_csv(output_csv_path, index=False)

print(f"Imputed data has been saved to: {output_csv_path}")
