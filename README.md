# LLM Powered RAG Application for finding GST Information #
<img width="1670" alt="Screenshot 2025-02-21 at 12 02 27 AM" src="https://github.com/user-attachments/assets/b4d32bf8-2268-4394-8df0-3c7a31cb616d" />

This project leverages the power of Retrival Augmented Generation (RAG) architecture to find the the gst rates of Indian goods and services, with only its product description. 

# Methodology 
The general overview of thsi model would be to first take the GST dataset, and perform EDA on top of it, to clean and retrieve the most valuable colomns from the dataset, optimising cost and performance. Once this information is found, this is then converted to a vector database using FAISS Encoding, of only a key-value pair like storage methodology. This process ensures that the data the can be read and understood by the llm. Post that a search query is executed to find the releavent results from the documents, once the correct Sr. No. (Key) is found, it is then pattern matched through the orignal database, to retrieve the full row information. s

## Data Cleaning and feeding 
There existed 2 issues with the orignal dataset. Both of them were fixed: 
* There existed Null and "Omitted" value rows, which needed to be cleaned
* Some values had spacing in the string literal.

The data feeding involved choosing only the Sr. No. and Description colomns of the dataset, this would make querying more efficient as no extra colomns are needed. This helps **save costs** and **increasing model accuracy**. 

## Vector Indexing 
This implementation creates a vector indexing system using the FAISS library to efficiently store and retrieve text-based data. The approach utilizes OpenAI embeddings to transform textual descriptions into vector representations, which are then indexed in FAISS for fast similarity searches. The goal of this encoding is to enable quick retrieval of relevant descriptions based on similarity, aiding in structured data search and analysis.

### Implementation Specifics
1. **Data Loading**: The script loads a CSV file containing product descriptions and serial numbers.
2. **Preprocessing**: It ensures that the required columns ("Description of Goods" and "S. No.") exist.
3. **Embedding Generation**: Each product description is converted into an embedding using OpenAI's embedding model.
4. **Vector Storage**: The embeddings, along with metadata (serial numbers), are stored in a FAISS vector store for efficient retrieval.
5. **Persistence**:
   - The FAISS index is saved locally for future use.
   - A metadata dictionary mapping descriptions to serial numbers is stored using pickle.

### Purpose of This Encoding
This encoding is designed to enable **efficient and scalable text-based search for structured data**. By converting textual descriptions into high-dimensional embeddings, the FAISS index allows for quick retrieval based on semantic similarity rather than exact keyword matching. This approach is particularly useful for B2B businesses handling a large volume of goods, ensuring that relevant items can be retrieved efficiently based on their descriptions.

## FAISS Retrieval
The FAISS retrival implementation enables efficient information retrieval using FAISS and OpenAI embeddings. The approach converts textual queries into vector representations and performs a similarity search against a pre-built FAISS index. The retrieved matches are cross-referenced with structured data in a CSV file to extract relevant details.

## User Interface
The UI is built in **Streamlit**, specifically for its easy to deploy nature and being quick to code. 

