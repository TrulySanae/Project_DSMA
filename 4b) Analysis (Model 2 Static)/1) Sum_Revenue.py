import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('/Users/sanaemessoudi/Desktop/DSMA/Final_Project_DSMA_Final/2c) Data_Model_2/4b) Data_with_Revenue.csv')

# Calculate the sum of the revenue columns for each row
df['total_revenue'] = df[['revenue_Monday', 'revenue_Tuesday', 'revenue_Wednesday',
                           'revenue_Thursday', 'revenue_Friday', 'revenue_Saturday',
                           'revenue_Sunday']].sum(axis=1)

# Drop the individual revenue columns
df.drop(columns=['revenue_Monday', 'revenue_Tuesday', 'revenue_Wednesday',
                 'revenue_Thursday', 'revenue_Friday', 'revenue_Saturday',
                 'revenue_Sunday'], inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv('1b) Sum_Revenue.csv', index=False)

# Display the updated DataFrame to check the result
print(df.head(10))