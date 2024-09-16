import os
import sys
import requests
import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import streamlit as st

nest_asyncio.apply()

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

encode_kwargs = {'normalize_embeddings': True}
embd_model = HuggingFaceInstructEmbeddings(model_name='intfloat/e5-large-v2',
                                           embed_instruction="",
                                           query_instruction="Represent this sentence for searching relevant passages: ",
                                           encode_kwargs=encode_kwargs)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

SAMBA_API_KEY = "" #PROVIDE GSAMBANOVA API KEY
SAMBA_API_URL = "" #PROVIDE YOUR SAMBANOVA API URL


def convert_memory_to_serializable(memory):
    chat_history = memory.load_memory_variables({})
    return [
        {"role": "user", "content": entry.content} if isinstance(entry, HumanMessage)
        else {"role": "assistant", "content": entry.content}
        for entry in chat_history.get("chat_history", [])
    ]


def query_sambanova(prompt, memory):
    headers = {
        'Authorization': f'Bearer {SAMBA_API_KEY}',
        'Content-Type': 'application/json',
    }

    formatted_memory = convert_memory_to_serializable(memory)

    data = {
        "model": "Meta-Llama-3.1-405B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            *formatted_memory,
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(SAMBA_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None


def process_url(url):
    loader = AsyncHtmlLoader([url], verify_ssl=False)
    docs = loader.load()

    html2text_transformer = Html2TextTransformer()
    docs = html2text_transformer.transform_documents(documents=docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", "."]
    )

    docs = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=docs, embedding=embd_model)

    return vectorstore, docs


def summarize_text(docs):
    summary_prompt = "Please summarize the following text:"
    summary = query_sambanova(summary_prompt + "\n\n".join([doc.page_content for doc in docs]), memory)
    return summary


st.title("Web data Extraction with SambaNova AI")

url_input = st.text_input("Enter a URL:", "")

if st.button("Summarize"):
    if url_input:
        st.write("### Processing...")
        vectorstore, docs = process_url(url_input)
        summary = summarize_text(docs)

        if summary:
            st.session_state.summary = summary
            st.session_state.vectorstore = vectorstore
            st.session_state.summarized = True
        else:
            st.session_state.summary = "No summary available."
            st.session_state.summarized = False

if st.session_state.get("summarized", False):
    st.write("### Summary:")
    st.write(st.session_state.summary)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write("### Conversation History:")
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"<p style='font-size:20px; font-weight:bold;'>You: {entry['content']}</p>",
                        unsafe_allow_html=True)
        elif entry["role"] == "assistant":
            st.markdown(f"<p style='font-size:20px;'>Assistant: {entry['content']}</p>", unsafe_allow_html=True)

    st.write("### Ask questions about the data:")
    user_question = st.text_area("Ask a question:", key="user_input")

    if st.button("Submit Question"):
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            vectorstore = st.session_state.vectorstore
            relevant_docs = vectorstore.similarity_search(user_question)
            relevant_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            answer = query_sambanova(user_question + "\n\n" + relevant_text, memory)

            if answer:
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                memory.save_context({"input": user_question}, {"output": answer})

