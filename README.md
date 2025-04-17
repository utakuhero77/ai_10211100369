THIERRY KOFI DAGBEY


DEPLOYED LINK 

https://mlproject-a7ysj2kkpjysjvpggat7z3.streamlit.app/



Instructions on How to Use  Features

1. Upload a PDF Document : Start by uploading a PDF document (e.g., `handbook.pdf`) that contains the knowledge base.
2. Processing the Document : The system will process the document, extract text, and create embeddings for efficient search.
3. Ask a Question: Enter your question in the text input box and click "Ask."
4. Receive an Answer: The system retrieves relevant information from the document and generates an answer using a pre-trained language model.


Description of Datasets and Models Used

Dataset: The uploaded PDF document serves as the knowledge base. For this demonstration, we use `handbook.pdf`.
Embedding Model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is used to create vector embeddings of text chunks.
-Language Model: google/flan-t5-large  https://huggingface.co/google/flan-t5-large is used to generate answers based on the retrieved text.


Architecture

1. Text Extraction: Extract text from the uploaded PDF using PyPDF.
2. Text Splitting: Split the extracted text into smaller chunks for efficient processing.
3. Embedding Creation**: Generate vector embeddings for each text chunk using the embedding model.
4. Vector Search: Store embeddings in Qdrant and perform similarity search to retrieve relevant chunks.
5. Answer Generation: Use the Hugging Face model to generate concise answers based on the retrieved chunks.

---

Detailed Methodology

1. Step *: Extract text from the uploaded PDF using PyPDF.
2. Step 2: Split the text into overlapping chunks of 500 characters with a 50-character overlap.
3. Step 3: Generate embeddings for each chunk using the all-MiniLM-L6-v2 model.
4. Step 4: Store the embeddings in Qdrant, a vector database optimized for similarity search.
5. Step 5: When a query is entered, encode it into a vector and perform a similarity search in Qdrant to retrieve the top 5 most relevant chunks.

6. Step 6: Combine the retrieved chunks and pass them to the google/flan-t5-large model to generate a concise answer.

