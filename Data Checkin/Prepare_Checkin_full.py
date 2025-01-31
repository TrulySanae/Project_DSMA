import csv
import json
import sys

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

# Input and output file paths
input_file = 'checkin_full.csv'
output_file = 'Prepared_Checkin_Full.csv'

# Open the input file and read the data
with open(input_file, 'r') as infile:
    reader = csv.DictReader(infile)

    # Prepare the output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write the header row
        writer.writerow(['business_id', 'date'])

        # Process each row in the input file
        for row in reader:
            # Parse the JSON string in column 'j'
            json_data = json.loads(row['j'])

            # Extract the business_id and dates
            business_id = json_data['business_id']
            dates = json_data['date']

            # Write the row with business_id and all dates as a single comma-separated string
            writer.writerow([business_id, dates])

print(f"Data has been processed and written to {output_file}")
