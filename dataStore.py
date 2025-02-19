from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
import pickle
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load CSV file
csv_path = "cleaned_data.csv"
df = pd.read_csv(csv_path)

# Ensure required columns exist
required_columns = ["Description of Goods", "S. No."]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Extract relevant columns
descriptions = df["Description of Goods"].astype(str).tolist()
serial_numbers = df["S. No."].astype(str).tolist()

# Initialize OpenAI Embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Convert each row into a LangChain Document
documents = [
    Document(page_content=desc, metadata={"S. No.": s_no})
    for desc, s_no in zip(descriptions, serial_numbers)
]

# Create FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Save FAISS index locally
faiss_save_path = "faiss_index"
vector_store.save_local(faiss_save_path)

# Save S. No. mappings in a pickle file
metadata_dict = {desc: s_no for desc, s_no in zip(descriptions, serial_numbers)}

with open("data_index.pkl", "wb") as f:
    pickle.dump(metadata_dict, f)

print("✅ FAISS index and metadata mapping saved successfully!")
