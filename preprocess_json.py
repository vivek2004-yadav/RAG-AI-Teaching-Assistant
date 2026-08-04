# ==========================================
# ORIGINAL OLLAMA CODE (COMMENTED OUT FOR COMPARISON)
# ==========================================
# import requests
# import os
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# 
# def create_embedding(text_list):
#     # Send request to local Ollama server running on port 11434
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })
#     embedding = r.json()["embeddings"] 
#     return embedding
# 
# jsons = os.listdir("jsons") 
# my_dicts = []
# chunk_id = 0
# 
# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
#        
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 
# 
# df = pd.DataFrame.from_records(my_dicts)
# joblib.dump(df, 'embeddings.joblib')
# ==========================================


# ==========================================
# NEW GEMINI API CODE (WITH LINE-BY-LINE COMMENTS)
# ==========================================
import requests  # Import requests library to make HTTP REST API calls to Google Gemini
import os  # Import os module to read file paths and environment variables
import json  # Import json module to parse and write JSON data
import numpy as np  # Import numpy for numerical/vector array processing
import pandas as pd  # Import pandas to organize data in tabular DataFrame format
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity to find semantic matches
import joblib  # Import joblib to save and load Python objects on disk
import time  # Import time module to add delays during retries and backoff
from dotenv import load_dotenv  # Import load_dotenv to load settings from configuration files

# Load Gemini API key from the user's backend configuration path
load_dotenv(r"c:\Users\enqui\Desktop\Placement\backend\.env")  
# Load Gemini API key from a local .env configuration file if present
load_dotenv()  

# Retrieve the Gemini API key from environment variables
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")  
# Check if API key is missing
if not api_key:  
    # Print error message if key was not found
    print("ERROR: GEMINI_API_KEY not found in environment.")  
    # Exit script execution immediately
    exit(1)  

# Define the new embedding function targeting the Google Gemini cloud API
def create_embedding(text_list):  
    # Replace empty/whitespace strings with a single space to avoid Gemini API errors
    sanitized_texts = [text if text.strip() else " " for text in text_list]  
    
    # Construct the Gemini REST API URL for batch embedding generation
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"  
    # Initialize empty list to accumulate generated embeddings
    embeddings = []  
    # Set the request batch size to 100 to stay within limits of the API
    batch_size = 100  
    
    # Loop over the text list in batches of 100
    for i in range(0, len(sanitized_texts), batch_size):  
        # Slice the list to retrieve the current batch of texts
        batch = sanitized_texts[i:i+batch_size]  
        # Initialize an empty list to build the request payload structure
        requests_payload = []  
        # Loop through each text in the current batch
        for text in batch:  
            # Format and append payload structure expected by Google's API
            requests_payload.append({  
                "model": "models/gemini-embedding-2",  
                "content": {  
                    "parts": [{"text": text}]  
                }  
            })  
            
        # Define maximum retry attempts to handle transient errors
        max_retries = 5  
        # Try to call the API, retrying up to max_retries times
        for attempt in range(1, max_retries + 1):  
            # Try block to catch failures and status codes
            try:  
                # Send the POST request to Google Gemini API
                response = requests.post(url, json={"requests": requests_payload}, timeout=60.0)  
                # If rate limit (429) is hit, print warning and wait
                if response.status_code == 429:  
                    # Notify user that rate limit occurred and delay is triggered
                    print(f"Rate limited (429). Retrying in 15 seconds... (Attempt {attempt}/{max_retries})")  
                    # Sleep execution for 15 seconds to cool down the rate limit
                    time.sleep(15)  
                    # Continue loop to next retry attempt
                    continue  
                # Raise an error if the status code indicates a failure
                response.raise_for_status()  
                # Parse the response payload into a JSON dictionary
                res_json = response.json()  
                # Loop through the list of generated embeddings returned by the API
                for emb in res_json.get("embeddings", []):  
                    # Append the vector float values to our embeddings list
                    embeddings.append(emb["values"])  
                # Break out of retry loop on success
                break  
            # Catch block for exceptions
            except Exception as e:  
                # Check if this was the last allowed retry attempt
                if attempt == max_retries:  
                    # Raise the error and abort if all retries failed
                    raise e  
                # Print a warning and retry status
                print(f"Error: {e}. Retrying in 10 seconds...")  
                # Sleep execution for 10 seconds before next attempt
                time.sleep(10)  
                
    # Return the list of generated embedding vectors
    return embeddings  


# Fetch all transcript json files from the jsons directory
jsons = [f for f in os.listdir("jsons") if f.endswith(".json")]  
# Initialize list to hold chunks and their vector embeddings
my_dicts = []  
# Initialize sequential ID tracking for index
chunk_id = 0  

# Loop through each transcript file name
for json_file in jsons:  
    # Open the JSON transcript file explicitly with UTF-8 encoding
    with open(f"jsons/{json_file}", encoding="utf-8") as f:  
        # Load JSON dictionary content
        content = json.load(f)  
    # Print status indicating embedding creation has started
    print(f"Creating Embeddings for {json_file}")  
    # Send all text chunks of this transcript to get embeddings
    embeddings = create_embedding([c['text'] for c in content['chunks']])  
       
    # Loop through each chunk of text in the loaded file
    for i, chunk in enumerate(content['chunks']):  
        # Assign unique sequential ID to chunk
        chunk['chunk_id'] = chunk_id  
        # Store the corresponding Gemini embedding vector
        chunk['embedding'] = embeddings[i]  
        # Increment sequential ID tracking
        chunk_id += 1  
        # Append chunk data to accumulator list
        my_dicts.append(chunk)  

# Build a Pandas DataFrame from the list of chunks
df = pd.DataFrame.from_records(my_dicts)  
# Save the final DataFrame as a Gemini-compatible embeddings database
joblib.dump(df, 'embeddings_gemini.joblib')  
# Print final completion status message
print(f"Success! Generated embeddings for {len(df)} chunks and saved to embeddings_gemini.joblib.")  



