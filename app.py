import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Konfigurasi
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_rag_docs"
OLLAMA_MODEL = "qwen2.5:1.5b" # Bisa diganti ke mistral atau model lain yang Anda pakai
EMBEDDING_MODEL = "BAAI/bge-m3"
st.set_page_config(page_title="Local RAG Chat", page_icon="🤖", layout="centered")
st.title("🤖 Local RAG Chat (Ollama + Qdrant)")

@st.cache_resource
def get_vectorstore():
    # Inisialisasi embeddings dan koneksi ke Qdrant
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    client = QdrantClient(url=QDRANT_URL)
    
    # Cek apakah collection ada
    if not client.collection_exists(COLLECTION_NAME):
        st.error(f"Collection '{COLLECTION_NAME}' belum ada di Qdrant. Silakan jalankan 'python ingest.py' terlebih dahulu.")
        st.stop()
        
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vectorstore

# Format dokumen untuk prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Setup Retriever & LLM
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",
    temperature=0
)

# Template Prompt
template = """Gunakan konteks berikut untuk menjawab pertanyaan di akhir. 
Jika kamu tidak tahu jawabannya, katakan saja bahwa kamu tidak tahu, jangan mencoba mengarang jawaban.

Konteks:
{context}

Pertanyaan: {question}

Jawaban:"""
prompt = ChatPromptTemplate.from_template(template)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Menyimpan riwayat obrolan (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari user
if user_input := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Menghasilkan respons
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Mencari jawaban dari dokumen..."):
            try:
                response = rag_chain.invoke(user_input)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
