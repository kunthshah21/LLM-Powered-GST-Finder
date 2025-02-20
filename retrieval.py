from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd

def retrieve_info_from_faiss(query, top_k=3):

    # 1. Load .env (to get OPENAI_API_KEY)
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file or pass the key directly.")

    # 2. Load CSV
    csv_path = "datasets/cleaned_data.csv"
    df = pd.read_csv(csv_path)

    # 3. Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 4. Load the FAISS index
    faiss_save_path = "datasets/faiss_index"
    vector_store = FAISS.load_local(
        faiss_save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 5. Run similarity search
    results = vector_store.similarity_search(query, k=top_k)

    # 6. Assemble the final info from the CSV
    output_rows = []
    for doc in results:
        s_no = doc.metadata.get("S. No.")
        description = doc.page_content

        matched_rows = df[df["S. No."].astype(str) == str(s_no)]
        if matched_rows.empty:
            continue

        for _, row in matched_rows.iterrows():
            row_dict = row.to_dict()
            output_rows.append({
                "S. No.": row_dict["S. No."],
                "Description of Goods": row_dict["Description of Goods"],
                "Full Row": row_dict
            })

    return output_rows