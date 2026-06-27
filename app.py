import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Help Desk")

st.caption(
    "Ask questions about company HR policies."
)

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

CORPUS_PATH = "/mount/src/zyro-dynamics-hr-corpus"

MODEL = "llama-3.3-70b-versatile"

REFUSAL = (
    "I can only answer questions using Zyro Dynamics HR policy documents."
)

# ------------------------------------------------
# LOAD RAG
# ------------------------------------------------

@st.cache_resource
def build_rag():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":5,
            "fetch_k":20,
            "lambda_mult":0.4
        }
    )

    llm = ChatGroq(
        model=MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
"""
You are Zyro Dynamics HR Help Desk.

Answer ONLY from the context.

If context does not contain the answer reply EXACTLY:

I can only answer questions using Zyro Dynamics HR policy documents.

Context:

{context}

Question:

{question}

Answer:
"""
    )

    return retriever,llm,prompt

retriever,llm,prompt=build_rag()

# ------------------------------------------------
# ASK BOT
# ------------------------------------------------

def ask_bot(question):

    docs=retriever.invoke(question)

    context="\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt=prompt.format(
        context=context,
        question=question
    )

    answer=llm.invoke(final_prompt).content

    if REFUSAL.lower() in answer.lower():

        return REFUSAL,[]

    sources=[]

    for doc in docs:

        sources.append(
            f"{os.path.basename(doc.metadata['source'])} | Page {doc.metadata['page']}"
        )

    return answer,sources

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.header("Sample Questions")

    st.write("- What is the leave policy?")
    st.write("- Can I work from home?")
    st.write("- How does performance review work?")
    st.write("- What is the travel reimbursement policy?")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages=[]

# ------------------------------------------------
# CHAT
# ------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages=[]

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "sources" in message:

            with st.expander("Sources"):

                for s in message["sources"]:

                    st.write(s)

question=st.chat_input("Ask an HR question...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    answer,sources=ask_bot(question)

    with st.chat_message("assistant"):

        st.markdown(answer)

        if sources:

            with st.expander("Sources"):

                for s in sources:

                    st.write(s)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer,
            "sources":sources
        }
    )