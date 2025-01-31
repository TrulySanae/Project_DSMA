
import pandas as pd
# Load the dataset
df = pd.read_csv(
    '3b) Data_No_Outliers.csv')


# Assuming the dataframe is called 'df'
# Example encoding for 'rating_price' column
def encode_rating_price(rating_price):
    if rating_price == 1:
        return 10  # Cheap
    elif rating_price == 2:
        return 15  # Somewhat cheap
    elif rating_price == 3:
        return 30  # Moderate
    elif rating_price == 4:
        return 40  # Expensive
    return 0  # Default to 0 for any other unexpected values

# Apply the encoding function
df['price_per_visitor'] = df['rating_price'].apply(encode_rating_price)

# List of day columns (visitor amounts for each day of the week)
days_of_week = ['Monday_visitors', 'Tuesday_visitors', 'Wednesday_visitors', 'Thursday_visitors',
                'Friday_visitors', 'Saturday_visitors', 'Sunday_visitors']

# Calculate the revenue for each day by multiplying visitor count with price per visitor
# Calculate revenue and name columns without "_visitors"
for day in days_of_week:
    day_name = day.replace("_visitors", "")  # Remove "_visitors" from column name
    df[f'revenue_{day_name}'] = df[day] * df['price_per_visitor']

# Drop the visitor amount columns after revenue calculation
df.drop(columns=days_of_week, inplace=True)

# Optionally, you can also drop 'price_per_visitor' if you don't need it anymore
df.drop(columns=['price_per_visitor', 'rating_price'], inplace=True)

# Save the updated DataFrame to a CSV file
output_file = '4b) Data_with_Revenue.csv'
df.to_csv(output_file, index=False)

# Print a confirmation message
print(f"File has been saved to {output_file}")
