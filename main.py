import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.qdrant import Qdrant
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient, models

def get_pdf_text(pdf_documents):
  text = ""
  for pdf in pdf_documents:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=800,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(embedding_model, collection_name):
    # url = f'{os.getenv("QDRANT_HOST")}:{os.getenv("QDRANT_PORT")}'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        # Collection exists
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # Collection does not exist; create it
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384,
                                               distance=models.Distance.COSINE))
        print(f"Collection '{collection_name}' created.")   

    return Qdrant(
        client=qdrant_client,
        collection_name=collection_name, 
        embeddings=embeddings)

def main():
    # Initialize variables
    load_dotenv()
    template = """Question: {question}

    Answer: Let's think step by step.
    """
    # prompt_template = PromptTemplate(template=template, input_variables=["question"])
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "google/flan-t5-xxl"
    
    # Setup GUI
    st.set_page_config(page_title='Chat Multiple PDFs',page_icon=':books:')
    
    if 'text' not in st.session_state:
        st.session_state.text = ""
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    collection_name = "my_collection"
    # Upload and process documents
    with st.sidebar:
        pdf_documents = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner('Processing...'):
                pdf_text = get_pdf_text(pdf_documents)
                text_chunks = get_chunks(pdf_text)
                vector_store=get_vectorstore(embedding_model=embedding_model,
                                   collection_name=collection_name)
                vector_store.add_texts(text_chunks)
                st.session_state.vector_store = vector_store

                st.success('Done! You can now ask questions to your PDFs')
    
    # Run questions
    st.title('Chat with Multiple PDFs :books:')
    user_question = st.text_input('Ask a question about the PDFs')
    if user_question is not None and user_question != '':
        retriever = st.session_state.vector_store.as_retriever()
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        llm = HuggingFaceHub(repo_id=llm_model, model_kwargs={"temperature":0.5, "max_length":512})

        qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                memory=memory)
    
        with st.spinner('Searching...'):
            if user_question:
                response = qa.run(user_question)
                st.write(response)
            else:
                st.warning('Please enter a question')
    
if __name__ == '__main__':
    main()