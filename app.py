import os
import streamlit as st
import pandas as pd
import pickle
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tiktoken
from fuzzywuzzy import process  # Fuzzy string matching

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load CSV data (for full row retrieval)
csv_path = "data.csv"
df = pd.read_csv(csv_path)

# Load FAISS index
faiss_load_path = "faiss_index"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vector_store = FAISS.load_local(
    faiss_load_path,
    embeddings,
    allow_dangerous_deserialization=True  # Enables safe pickle loading
)

retriever = vector_store.as_retriever()

# Load "Description of Goods" → "S. No." mapping from pickle
with open("data_index.pkl", "rb") as f:
    metadata_dict = pickle.load(f)

# Initialize OpenAI GPT-4o Mini (Chat Model)
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
    openai_api_key=openai_api_key
)

# Token counter function
enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text):
    return len(enc.encode(text))


# Define the optimized prompt template
prompt_template = PromptTemplate(
    input_variables=["product", "retrieved_descriptions"],
    template="""You are a GST database assistant. 
    You must find the best match for the product description provided by the user. 

    The user is searching for: "{product}". 

    Here are the closest matches from the GST dataset: 
    {retrieved_descriptions}

    From these options, return the **three closest** product descriptions, separated by commas. 
    Do not make up new products—ONLY choose from the given dataset options."""
)


# Fuzzy matching function (for better retrieval accuracy)
def fuzzy_match(query, choices, threshold=80):
    match, score = process.extractOne(query, choices)
    return match if score >= threshold else None


# Streamlit UI
st.title('GST Rate Retriever')

query = st.text_input("Give Product Description")

if query:
    st.text("Processing your query...")

    # Debugging: Count input tokens
    input_tokens = count_tokens(query)
    print(f"🔹 Query Input Tokens: {input_tokens}")

    # Step 1: Retrieve the most similar documents from FAISS
    retrieved_docs = retriever.get_relevant_documents(query, k=5)  # Get top 5 matches for better accuracy
    retrieved_descriptions = [doc.page_content for doc in retrieved_docs]

    # Debugging: Print retrieved descriptions
    print(f"🔍 Retrieved Descriptions: {retrieved_descriptions}")

    if not retrieved_descriptions:
        st.error("❌ No relevant results found in FAISS index.")
        print("❌ No relevant descriptions found in FAISS.")
    else:
        # Step 2: Format retrieved descriptions for LLM prompt
        retrieved_text = "\n".join(retrieved_descriptions)

        # Step 3: Run LLM with refined prompt and dataset context
        prompt = prompt_template.format(product=query, retrieved_descriptions=retrieved_text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that retrieves GST data."},
            {"role": "user", "content": prompt}
        ]

        llm_response = llm(messages).content

        # Debugging: Show raw LLM response
        print(f"🔍 LLM Response: {llm_response}")

        # Step 4: Extract top 3 descriptions from LLM output
        top_descriptions = [desc.strip() for desc in llm_response.split(",")]

        # Step 5: Retrieve corresponding S. No. values
        s_no_list = []
        for desc in top_descriptions:
            s_no = metadata_dict.get(desc, None)

            # If no exact match, use fuzzy matching
            if not s_no:
                fuzzy_match_desc = fuzzy_match(desc, metadata_dict.keys())
                if fuzzy_match_desc:
                    s_no = metadata_dict.get(fuzzy_match_desc)

            if s_no:
                s_no_list.append(s_no)

        # Debugging: Show retrieved S. No. values
        print(f"🔍 Retrieved S. No. List: {s_no_list}")

        # Ensure at least one result is returned
        if not s_no_list:
            s_no_list = list(metadata_dict.values())[:3]  # Fallback: Pick any 3 S. No.

        # Step 6: Fetch the top 3 matching rows from CSV
        matched_rows = df[df["S. No."].astype(str).isin(s_no_list)]

        if not matched_rows.empty:
            # Reset index for display purposes
            matched_rows.reset_index(drop=True, inplace=True)

            # Debugging: Count output tokens
            output_tokens = count_tokens(matched_rows.to_string(index=False))
            print(f"🔹 Output Tokens Used: {output_tokens}")

            # Display results dynamically in Streamlit with numbering
            st.header("GST Rate Details (Top Matches)")

            for idx, row in matched_rows.iterrows():
                st.markdown(f"### {idx + 1}:")
                st.dataframe(pd.DataFrame([row]))

        else:
            st.error("❌ No relevant GST entries found.")

        print("✅ Query Process Completed.")


