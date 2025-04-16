import os
import pandas as pd

input_dir = "C:/Users/paulk/Desktop/DataLakes/Practical Work1/random_split/dev"
output_file = "C:/Users/paulk/Desktop/DataLakes/Practical Work1/combined_output.csv"

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.

    """

    # Step 1: Initialize an empty list to store DataFrames
    dataframes = []

    # Step 2: Loop over files in the input directory
    for path, dirs, files in os.walk(input_dir):
        for filename in files:
            # Step 3: Check if the file is a CSV
            if filename.endswith(".csv"):
                file_path = os.path.join(path, filename)  # Get the full path to the file
                # Step 4: Read the CSV file using pandas
                df = pd.read_csv(file_path)
                # Step 5: Append the DataFrame to the list
                dataframes.append(df)

    # Step 6: Concatenate all DataFrames
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Step 7: Save the combined DataFrame to output_file
    combined_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    unpack_data(input_dir, output_file)
