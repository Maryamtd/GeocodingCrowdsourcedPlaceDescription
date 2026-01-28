
import json
import pandas as pd
import os
import re

# Define paths
batch_dir = "AI_review/batches/task1_StateNocoNoci"
output_dir = "AI_review/outputs/task1_StateNocoNoci"
csv_output_dir = "AI_review/outputs/task1_StateNocoNoci/csvs"

# Create the output directory if it doesn't exist
os.makedirs(csv_output_dir, exist_ok=True)

# Function to get file number
def get_file_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

# Get list of files in input and output directories
input_files = sorted([f for f in os.listdir(batch_dir) if get_file_number(f) is not None], key=get_file_number)
output_files = sorted([f for f in os.listdir(output_dir) if get_file_number(f) is not None], key=get_file_number)

# Process each pair of input and output files
for input_file, output_file in zip(input_files, output_files):
    if input_file.endswith('.jsonl') and output_file.endswith('.jsonl'):
        # Read input batch JSONL file
        input_data = []
        with open(os.path.join(batch_dir, input_file), 'r') as file:
            for line in file:
                input_data.append(json.loads(line.strip()))

        # Extract 'custom_id' and 'content' from the input data
        input_records = []
        for item in input_data:
            custom_id = item['custom_id']
            stdbirthplace = item['body']['messages'][1]['content']
            input_records.append({'custom_id': custom_id, 'stdbirthplace': stdbirthplace})

        input_df = pd.DataFrame(input_records)

        # Read output batch JSONL file
        output_data = []
        with open(os.path.join(output_dir, output_file), 'r') as file:
            for line in file:
                output_data.append(json.loads(line.strip()))

        # check state only _ only for task 101
        output_records = []
        for item in output_data:
            custom_id = item['custom_id']
            response_body = item['response']['body']['choices'][0]['message']['content']
            response_content = json.loads(response_body)
            state = response_content.get('state', None)
            ambiguity = response_content.get('ambiguity', None)
            output_records.append({
                'custom_id': custom_id,
                'state': state,
                'ambiguity': ambiguity
            })

        output_df = pd.DataFrame(output_records)

        # Debugging: Print first few rows to verify extraction
        #print(output_df.head())

        # Merge DataFrames on 'custom_id'
        merged_df = pd.merge(input_df, output_df, on='custom_id')

        # Save to CSV
        csv_file_name = f"combined_output_{get_file_number(input_file)}.csv"
        merged_df.to_csv(os.path.join(csv_output_dir, csv_file_name), index=False)

        print(f"Combined CSV file saved to {csv_output_dir}/{csv_file_name}")

