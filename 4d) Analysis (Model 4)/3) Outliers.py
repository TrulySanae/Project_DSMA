# Import necessary libraries
import pandas as pd
from feature_engine.outliers import Winsorizer


# Read the data from CSV
df = pd.read_csv('1b) Add_Checkin_Data.csv')


# Initialize the Winsorizer
winsorizer = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=["week_hours", "rating_review_count", "rating_popularity",
               "Monday_visitors", "Tuesday_visitors", "Wednesday_visitors", "Thursday_visitors",
               "Friday_visitors", "Saturday_visitors", "Sunday_visitors"]
)

# Apply the Winsorizer transformation
df = winsorizer.fit_transform(df)

# Show the transformed DataFrame
print(df.head(10))
# Save the transformed DataFrame to a new CSV file
df.to_csv('3b) Data_No_Outliers.csv', index=False)

print("Transformed data saved to CSV!")
