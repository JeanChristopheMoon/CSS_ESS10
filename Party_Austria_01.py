import pandas as pd

# List of columns to extract
party_vote_columns = [
    "prtvtdat"
]

# Path to your input file
input_file = "/Users/x/Desktop/LLmScript/RandomForest/ESS11.csv"

# Path for the output file
output_file = "/Users/x/Desktop/LLmScript/RandomForest/partyـAustria_01.csv"

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


################################Converting to 0 or 1


# Load your CSV file
df = pd.read_csv('/Users/x/Desktop/LLmScript/RandomForest/partyـAustria_01.csv')  # replace with your actual file name

# Recode 'prtvtdat': 1 if FPÖ (value 3), 0 if other valid parties (1–8 except 3), else missing
df['fpo_vote'] = df['prtvtdat'].apply(lambda x: 1 if x == 3 else (0 if x in range(1, 9) else pd.NA))

# Save to a new CSV file
df.to_csv('output_fpo_vote.csv', index=False)


