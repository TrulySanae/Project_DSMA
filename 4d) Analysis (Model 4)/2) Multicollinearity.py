import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def check_multicollinearity(csv_file, columns_to_check):
    """
    This function checks for multicollinearity in the specified columns of a CSV file
    and suggests variables to exclude based on Variance Inflation Factor (VIF).

    Args:
    - csv_file (str): Path to the CSV file.
    - columns_to_check (list): List of column names to check for multicollinearity.

    Returns:
    - pd.DataFrame: DataFrame showing each column and its VIF score.
    - list: List of columns suggested for exclusion.
    """
    # Load the data with error handling
    df = pd.read_csv(csv_file, on_bad_lines='skip')  # Skip bad lines if needed

    # Filter the data to include only the specified columns
    df_subset = df[columns_to_check].dropna()

    # Add a constant column for VIF calculation
    df_subset = sm.add_constant(df_subset)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_subset.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_subset.values, i) for i in range(df_subset.shape[1])
    ]

    # Drop the constant column from results
    vif_data = vif_data[vif_data["Feature"] != "const"]

    # Suggest variables to exclude (VIF > 10)
    exclude_columns = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()

    return vif_data, exclude_columns


# Define the columns to check and the CSV file path
csv_file_path = "1b) Add_Checkin_Data.csv"
columns_to_check = ["week_hours", "rating_stars", "rating_review_count", "rating_popularity"]

# Call the function and display results
vif_results, columns_to_exclude = check_multicollinearity(csv_file_path, columns_to_check)

print("VIF Results:")
print(vif_results)

if columns_to_exclude:
    print("\nSuggested columns to exclude due to high multicollinearity:")
    print(columns_to_exclude)
else:
    print("\nNo columns have significant multicollinearity.")
