import streamlit as st
import pandas as pd
from retrieval import retrieve_info_from_faiss

# Streamlit UI
st.set_page_config(page_title="GST Retrieval System", layout="wide")

st.title("📜 GST Retrieval System")
st.markdown("#### 🔎 Search for product descriptions and retrieve relevant GST details.")

# Input box for user query
user_query = st.text_input("🔍 Enter product description:", "")

# Search button
if st.button("Search"):
    if user_query.strip():  # Ensure input is not empty
        with st.spinner("Fetching results..."):
            results = retrieve_info_from_faiss(user_query, top_k=3)

        if results:
            st.success(f"✅ Found {len(results)} matching results:")

            # Loop through each retrieved row and display as a table
            for i, ans in enumerate(results, start=1):
                st.markdown(f"### 🏷 Result {i}")

                # Convert dictionary row into a DataFrame for table display
                row_df = pd.DataFrame([ans["Full Row"]])
                st.dataframe(row_df.style.set_properties(**{'text-align': 'left'}))

                st.markdown("---")  # Separator line
        else:
            st.warning("⚠️ No matching results found.")
    else:
        st.error("❌ Please enter a product description.")
