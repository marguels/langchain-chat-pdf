from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate

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
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_embeddings(text_chunks, model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    #TODO: Use Qdrant, Chroma or Pinecone to persist database
    return vector_store

def main():
    # Initialize variables
    load_dotenv()
    template = """Question: {question}

    Answer: Let's think step by step.
    """
    prompt_template = PromptTemplate(template=template, input_variables=["question"])
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embedding_model = "hkunlp/instructor-xl"
    llm_model = "google/flan-t5-xxl"
    
    # Setup GUI
    st.set_page_config(page_title='Chat Multiple PDFs',page_icon=':books:')
    
    if 'text' not in st.session_state:
        st.session_state.text = ""
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # Upload and process documents
    with st.sidebar:
        pdf_documents = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner('Processing...'):
                pdf_text = get_pdf_text(pdf_documents)
                text_chunks = get_chunks(pdf_text)
                st.session_state.vector_store = create_embeddings(text_chunks=text_chunks, model_name=embedding_model)
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