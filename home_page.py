import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
def load_response(query):
    try:
        # Load the document using PyMuPDFLoader
        loader = PyMuPDFLoader('uploaded_file.pdf')
        data = loader.load()
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(["\n", ".", "!", "?"], chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(data)
        # Initialize embeddings using Google Generative AI
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Create vector index using FAISS with document chunks and embeddings
        vector_index = FAISS.from_documents(chunks, embedding=embeddings)
        # Initialize QA components
        llm = ChatGroq(temperature=0.5, model='llama3-70b-8192')
        retriever = vector_index.as_retriever()
        memory = ConversationSummaryMemory(llm=llm)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='map_reduce', retriever=retriever, memory=memory)
        message = qa(query)
        return message['result']
    except Exception as e:
        print(f"Error occurred: {e}")
# Set up page configuration
st.set_page_config(page_title='Upload your PDF')
# Define a function for the second page
def second_page():
    st.title("QA over your PDF file 🤓")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if query := st.chat_input("What is your question?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        response = load_response(query)
    # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
def upload_page():
    st.title('Load your pdf file')
    uploaded_file = st.file_uploader(label='Upload your PDF file', type=['pdf'], label_visibility="visible")
    if uploaded_file:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file = uploaded_file
        st.success("File uploaded and saved successfully!")
        time.sleep(2)
        st.session_state.page = 'second_page'
        st.experimental_rerun()
    else:
        st.warning("Please upload a PDF file.")

# Check the session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'upload_page'

if st.session_state.page == 'upload_page':
    upload_page()
elif st.session_state.page == 'second_page':
    second_page()
