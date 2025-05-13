import pandas as pd

# List of columns to extract
party_vote_columns = [
    "prtvtdat", "prtvtebe", "prtvtchr", "prtvtccy", "prtvtffi", 
    "prtvtffr", "prtvgde1", "prtvgde2", "prtvtegr", "prtvthhu", 
    "prtvteis", "prtvteie", "prtvteit", "prtvclt1", "prtvclt2", 
    "prtvclt3", "prtvtinl", "prtvtcno", "prtvtfpl", "prtvtept", 
    "prtvtbrs", "prtvtesk", "prtvtgsi", "prtvtges", "prtvtdse", 
    "prtvthch", "prtvtdgb"
]

# Path to your input file
input_file = "your_input_file.csv"  # Replace with your actual input file path

# Path for the output file
output_file = "party_voting_data.csv"  # You can change this name if you prefer

# Read the CSV file
try:
    # First check if the file exists and if all columns are present
    df = pd.read_csv(input_file)
    
    # Check which of the requested columns are actually in the dataset
    available_columns = [col for col in party_vote_columns if col in df.columns]
    missing_columns = [col for col in party_vote_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following requested columns were not found in the dataset: {', '.join(missing_columns)}")
    
    if not available_columns:
        print("Error: None of the requested columns were found in the dataset.")
        exit()
    
    # Extract only the available columns that we want
    extracted_data = df[available_columns]
    
    # Save to a new CSV file
    extracted_data.to_csv(output_file, index=False)
    
    print(f"Successfully extracted {len(available_columns)} party voting columns to {output_file}")
    print(f"Columns extracted: {', '.join(available_columns)}")

except FileNotFoundError:
    print(f"Error: Could not find the file {input_file}")
except Exception as e:
    print(f"An error occurred: {e}")
