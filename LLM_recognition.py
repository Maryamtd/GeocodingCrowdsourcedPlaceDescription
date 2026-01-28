''' This script is used to use OpenAI API in toponym resolution.'''

import json
import logging
import time
from openai import OpenAI
import pandas as pd
import os


# Initialize logging
logging.basicConfig(filename='AI_review/logfile.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Initializing OpenAI client with the API key
client = OpenAI()

places_path = "AI_review/input/task1_StateNocoNoci.csv" # Change the CSV file name for other tasks
df = pd.read_csv(places_path)
logging.info("Loaded CSV file successfully")
print (df.head())

system_prompt = '''
You will be provided with unstructured birthplace string, which may include typos, abbreviations, extra words, historical or contemporary names and geographical features.
Your task is to process birthplace string step-by-step to provide the most accurate identification of the location at the finest available level (state, county, and city/township/village/populated place) based on historical and geographical records.
Your response must follow these guidelines:\n
1. Correct spelling errors and resolve typos and non-standard abbreviations while maintaining accuracy.\n
2. Identify and extract the state from the birthplace string.\n
3. Remove State: Isolate the remaining string by removing the state name.\n
4. Determine if the remaining components (county or city/township/village/populated place) correspond to actual locations in historical or contemporary geographical records.\n
5. Confirm whether the identifying county or city/township/village/populated place corresponds to a county or populated place within the identified state.\n
6. If the birthplace does not correspond to a valid place in the state, determine if the remaining string refers to a valid location in a different state.\n
7. Confirm whether a place with the corrected name has ever existed.
output a JSON object in lowercase with the following structure:\n
{
    state: correct state name,
    county: correct county,
    city: correct city/township/ village/ populated place,
    exists: 'yes' or 'no' // whether the place have ever existed
}
Important notes:\n
- Do not "guess" or "hallucinate" information.\n
- If a birthplace cannot be resolved accurately or confidently, pass it without attempting to generate an incorrect or speculative output.\n
- If the birthplace is ambiguous or it is a place name outside of united states do not perform any steps and pass.\n
- Include only names (no additional labels like "parish", "county" or "city").\n
- do not include punctuation.\n
- Include the city field only if explicitly mentioned in the birthplace; if only the county and state are provided, leave the city field blank.\n
Examples:\n
Input 1: 'he was born in north hero virginia'
Output 1: {'north hero', 'grand isle', 'vermont'}\n
Input 2: 'calpeper co va'
Output 2: {'' , 'culpeper', 'virginia'}\n
Input 3: 'decuir settlement marksvilleavoyelles parish louisiana'
Output 3: {'marksville', 'avoyelles', 'louisiana'}\n
'''

def create_and_submit_batch(tasks, batch_index):
    directory_path = "AI_review/batches/task1_StateNocoNoci" # Change the directory path for other tasks
    os.makedirs(directory_path, exist_ok=True)
    file_name = os.path.join(directory_path, f"batch_tasks_{batch_index}.jsonl")

    try:
        with open(file_name, 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')
        logging.info(f"Batch tasks file {batch_index} created successfully")
    except Exception as e:
        logging.error(f"Error creating tasks file {batch_index}: {e}")
        raise

    try:
        batch_file = client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        logging.info(f"Batch file {batch_index} created successfully")
    except Exception as e:
        logging.error(f"Error creating batch file {batch_index}: {e}")
        raise

    try:
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logging.info(f"Batch job {batch_index} created successfully")
    except Exception as e:
        logging.error(f"Error creating batch job {batch_index}: {e}")
        raise

    try:
        # Polling until the status is 'completed'
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            logging.info(f"Batch Job {batch_index} Status: {batch_job.status}")
            if batch_job.status == 'completed':
                break
            elif batch_job.status == 'failed':
                logging.error(f"Batch job {batch_index} failed.")
                raise RuntimeError(f"Batch job {batch_index} failed.")
            time.sleep(1800)  # Wait for 30 min seconds before checking again
    except Exception as e:
        logging.error(f"Error during batch job {batch_index} polling: {e}")
        raise

    # Retrieving the result file id
    try:
        result_file_id = batch_job.output_file_id
        if result_file_id:
            result = client.files.content(result_file_id).content
            result_directory_path ="AI_review/output/task1_StateNocoNoci" # Change the directory path for other tasks
            os.makedirs(result_directory_path, exist_ok=True)
            result_file_name = os.path.join(result_directory_path, f"batch_{batch_index}_reviewed.jsonl")
            with open(result_file_name, 'wb') as file:
                file.write(result)
            logging.info(f"Result file {batch_index} written successfully")
        else:
            logging.error(f"No output file ID found for batch job {batch_index}.")
            raise ValueError(f"No output file ID found for batch job {batch_index}.")
    except Exception as e:
        logging.error(f"Error retrieving or writing result file {batch_index}: {e}")
        raise

# Batch processing
# Creating an array of json tasks
# Splitting into smaller batches to avoid timeouts
batch_size = 200
tasks = []
for index, row in df.iterrows():
    place = row['stdbirthplace']

    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o",
            "temperature": 0,
            "response_format": {
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": place
                }
            ],
        }
    }

    tasks.append(task)

    if len(tasks) == batch_size:
        create_and_submit_batch(tasks, index // batch_size)
        tasks = []  # Reset the tasks list for the next batch

# Submit the remaining tasks if any
if tasks:
    create_and_submit_batch(tasks, (index // batch_size) + 1)
