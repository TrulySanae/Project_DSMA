import pandas as pd
from datetime import datetime


def merge_csv_files(file1, file2, output_file):
    """
    Merges two CSV files on the 'business_id' column and writes the result to a new file.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path to save the merged CSV file.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    # Read the CSV files into dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge the dataframes on the "business_id" column
    merged_df = pd.merge(df1, df2, on='business_id', how='inner')

    # Drop the 'rating' and 'business_id' columns from the merged dataframe
    merged_df = merged_df.drop(columns=['business_id'])

    # Write the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")

    return merged_df


# Parse check-in dates into weekday counts
def parse_checkins_to_weekdays(check_in_dates):
    """
    Parses a list of check-in dates (with multiple dates in a single cell) and converts them to weekday counts.

    Args:
        check_in_dates (pd.Series): A series of check-in dates in the format 'YYYY-MM-DD HH:MM:SS,...'.

    Returns:
        pd.DataFrame: A dataframe with columns for each weekday and their respective counts.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_data = []

    for cell in check_in_dates:
        weekday_counts = {weekday: 0 for weekday in weekdays}  # Initialize counts for each weekday

        if pd.notna(cell):  # Ensure the cell is not NaN
            # Split the cell into individual dates
            dates = cell.split(',')
            for date in dates:
                # Strip whitespace and extract the date part only (before the time, if present)
                date_part = date.strip().split(' ')[0]
                try:
                    weekday = datetime.strptime(date_part, "%Y-%m-%d").strftime('%A')
                    if weekday in weekday_counts:
                        weekday_counts[weekday] += 1
                except ValueError:
                    print(f"Invalid date format: {date}")

        weekday_data.append(weekday_counts)

    return pd.DataFrame(weekday_data)


# Example usage
if __name__ == "__main__":
    # File paths
    file1 = '../Data Checkin/Prepared_Checkin_Full.csv'
    file2 = '../2) Data Cleaning/6b) Imputed_Dataset.csv'
    output_file = '1b) Add_Checkin_Data.csv'  # Name of the output file

    # Merge the files
    merged_df = merge_csv_files(file1, file2, output_file)

    # Ensure 'check_in_dates' column exists before processing
    if 'date' in merged_df.columns:
        # Parse check-in dates into weekdays and get the counts for each weekday
        weekday_visits_df = parse_checkins_to_weekdays(merged_df['date'])

        # Combine the weekday visit counts with the merged dataframe
        merged_df = pd.concat([merged_df, weekday_visits_df], axis=1)
        merged_df = merged_df.drop(columns=['date'])
        # Save the updated dataframe with weekday visits

        # Columns to rename
        columns_to_rename = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        merged_df.rename(columns={col: f"{col}_visitors" for col in columns_to_rename}, inplace=True)

        merged_df.to_csv(output_file, index=False)
        print("Weekday visits added and updated file saved.")
    else:
        print("Column 'date' not found in the merged dataframe.")
