
import os, json, requests
from dotenv import load_dotenv
import chromadb
from huggingface_hub import InferenceClient

# Load API key from .env file
load_dotenv()
READ_API_KEY= os.getenv("READ_API_KEY")

# Initialize Hugging Face Inference Client
client = InferenceClient(
    api_key=os.environ["READ_API_KEY"],
)

# Set model name
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita"

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# Get user query
user_query = input ("Enter your question about growing vegetables: ")

# Query ChromaDB for relevant document chunks based on user query
results = collection.query(
    query_texts = [user_query],
    n_results = 4
)

# Display the top 4 relevant document chunks
print("\nTop 4 relevant document chunks:\n")
for i,doc in enumerate(results['documents'][0]):
    print(f"{i+1}.{doc}\n{'-'*50}\n")

# Create system prompt for the growing vegetables model using the retrieved document chunks and user query
system_prompt = f"""
You are a helpful assistant. You answer questions about growing 
vegetables in Florida. Only answer based on the knowledge provided 
below. Don't make things up. If you don't know the answer, just say: I 
don't know.
-------------------
Data :
{results['documents'][0]}
"""
prompt = f"{system_prompt}\n\nQuestion: {user_query}"



# Prepare data for Hugging Face Inference API request
data = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ],
    "temperature": 0.7,
    "max_tokens": 300
}
# Make the API request to get the model's response
response = client.chat.completions.create(**data)

# Collecting response
full_text = response.choices[0].message.content.strip()

# Print Output in JSON format
print("\nAnswer:\n")
output_json = {
    
    "question": user_query,
    "model_name": MODEL_NAME,
    "model_response": full_text
}

print(json.dumps(output_json, ensure_ascii=False, indent=2))
