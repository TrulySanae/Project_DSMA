import csv
import os

def split_csv(input_file, output_dir, rows_per_file=31500, skip_rows=31574):
    """
    Splits a CSV file into multiple smaller CSV files.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the output CSV files.
        rows_per_file (int): Number of rows per output file.
        skip_rows (int): Number of rows to skip at the beginning of the input file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        # Extract the header
        header = next(reader)

        # Skip the first `skip_rows` rows
        for _ in range(skip_rows - 1):
            next(reader, None)

        # Initialize variables
        file_count = 0
        current_file_rows = []

        for i, row in enumerate(reader, start=1):
            current_file_rows.append(row)

            # When the row count reaches the limit, write to a new file
            if i % rows_per_file == 0:
                output_file = os.path.join(output_dir, f'output_part_{file_count + 1}.csv')
                with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
                    writer = csv.writer(output_csv)
                    writer.writerow(header)  # Write the header
                    writer.writerows(current_file_rows)
                print(f"Written {len(current_file_rows)} rows to {output_file}")

                # Reset for the next file
                file_count += 1
                current_file_rows = []

        # Write any remaining rows to the last file
        if current_file_rows:
            output_file = os.path.join(output_dir, f'output_part_{file_count + 1}.csv')
            with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
                writer = csv.writer(output_csv)
                writer.writerow(header)  # Write the header
                writer.writerows(current_file_rows)
            print(f"Written {len(current_file_rows)} rows to {output_file}")

if __name__ == "__main__":
    input_csv_path = "3b) Deleted_Non_Restaurant_Rows.csv"
    output_directory = "./4b) Distributed_Files/"
    split_csv(input_csv_path, output_directory)
