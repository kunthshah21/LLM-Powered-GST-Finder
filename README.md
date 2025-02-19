# LLM Powered RAG Application for finding GST Information #
This project leverages the power of Retrival Augmented Generation (RAG) architecture to find the the gst rates of Indian goods and services, with only its product description. 

# Methodology #
The general overview of thsi model would be to first take the GST dataset, and perform EDA on top of it, to clean and retrieve the most valuable colomns from the dataset, optimising cost and performance. Once this information is found, this is then converted to a vector database using FAISS Encoding. 
This process ensures that the data the can be read and understood by the llm. This would then be followed by an llm query to retrieve the top matching results, this would give the output as a json document. This processs of llm querying is repeated 10 times, to create a scoring like system to achieve the highest results of getting the correct gst information. This information would then be used to fetch the Sr. No. of the goods, this would then be pattern matched from the original csv based dataset and the correct rows would be identified and retrieved. 


## Data Cleaning and Transformation ##

