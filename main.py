import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def fetch_text(retrieved_vectors):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_vectors)
    return context_text

def chains(query):

    embedding= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # Check if the index exists before loading
    if os.path.exists("WEB_RESEARCH_PROJECT/faiss_index"):
        vectorstore = FAISS.load_local(
            "WEB_RESEARCH_PROJECT/faiss_index", 
            embeddings= embedding, 
            allow_dangerous_deserialization=True # Required for loading pickle files
        )
    else:
        return st.error("Please load URLs and process them first!")

    simple_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 5}
    )

    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite-001", temperature = 0.5)

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever= simple_retriever,
        base_compressor= compressor
    )

    parallel_chain = RunnableParallel({
        "context" : compression_retriever | RunnableLambda(fetch_text),
        "question" : RunnablePassthrough()
    })

    parser = StrOutputParser()

    prompt = PromptTemplate(
        template = """
        You are a very intelligent reserch Expert,
        you have to reserch on the user question using the context which is provided to you and answer the user queries

        context : {context}

        question : {question}
        """,
        input_variables = ["context", "question"]
    )

    final_chain = parallel_chain | prompt | llm | parser

    return final_chain.invoke(query)

st.title("Research Tool")

with st.sidebar:

    st.subheader("URLs of News")

    urls = []

    if "url_count" not in st.session_state:
        st.session_state.url_count = 1
    if "processed" not in st.session_state:
        st.session_state.processed = False

    def add_url():
        st.session_state.url_count +=1
        return st.session_state.url_count
    
    def remove_url():
        if st.session_state.url_count > 1:
            st.session_state.url_count -= 1
        return st.session_state.url_count
    
    col1, col2 = st.columns(2)

    with col1:
        st.button("Add URL", on_click=add_url)
    with col2:
        st.button("Remove URL", on_click= remove_url)


    for i in range(st.session_state.url_count):
        url = st.text_input(f"URL {i+1}")
        urls.append(url)


    load_urls = st.button("Load URLs")

    if load_urls:

        # filter the empty input
        valid_urls = [url for url in urls if url.strip()]

        if not valid_urls:
            st.error("Please enter at least one URL.!!!")
        else: 
            with st.spinner("Processing..."):
                # load the data
                url_loader = UnstructuredURLLoader(urls = urls)
                data = url_loader.load()

                # split the data
                splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 30)
                chunks = splitter.split_documents(data)

                # create vectorstore
                vectorstore = FAISS.from_documents(
                    documents= chunks,
                    embedding= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
                )

                # save vectorstore in local
                if not os.path.exists("WEB_RESEARCH_PROJECT"):
                    os.makedirs("WEB_RESEARCH_PROJECT")
                vectorstore.save_local("WEB_RESEARCH_PROJECT/faiss_index")

                st.session_state.processed = True
                st.success("Process Has been DONE")
# shows the question box only if we saved index or processed any one
if st.session_state or os.path.exists("WEB_RESEARCH_PROJECT/faiss_index"):
    query = st.text_input("Question: ")

    if query: 
        with st.spinner("Searching for answer..."):
            result = chains(query)
            st.write(result)
else:
    st.info("Please add URLs and Click on 'Load URLs' Button")



        
