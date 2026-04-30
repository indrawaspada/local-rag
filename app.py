import os
import streamlit as st
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from evaluator import run_ragas_evaluation

# Konfigurasi
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_rag_docs"
OLLAMA_MODEL = "qwen2.5:1.5b" # Bisa diganti ke mistral atau model lain yang Anda pakai
EMBEDDING_MODEL = "BAAI/bge-m3"

st.set_page_config(page_title="Local RAG Chat & Eval", page_icon="🤖", layout="wide")

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    client = QdrantClient(url=QDRANT_URL)
    
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Dinaikkan ke 5 untuk akurasi lebih baik

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

# --- UI Layout ---
tab_chat, tab_eval = st.tabs(["💬 Chat dengan SOP", "📊 Evaluasi Ragas"])

with tab_chat:
    st.title("💬 Chat dengan Dokumen SOP")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Tanyakan sesuatu tentang SOP Anda..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Mencari jawaban dari dokumen..."):
                try:
                    response = rag_chain.invoke(user_input)
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

with tab_eval:
    st.title("📊 Dashboard Evaluasi Ragas")
    st.write("Gunakan tab ini untuk mengukur kualitas jawaban sistem menggunakan Google Gemini sebagai juri.")

    # Daftar Pertanyaan Tes (Ground Truth) - Bisa ditambah sendiri oleh user
    test_questions = [
        {"q": "Bagaimana prosedur pengajuan cuti akademik?", "gt": "Mahasiswa mengajukan permohonan tertulis kepada Dekan melalui Ketua Program Studi dengan melampirkan berkas pendukung."},
        {"q": "Apa syarat permohonan izin aktif setelah cuti?", "gt": "Mahasiswa harus melampirkan SK Rektor tentang izin cuti akademik dan surat keterangan sehat jika cuti karena sakit."},
        {"q": "Siapa yang berhak mendapatkan beasiswa?", "gt": "Mahasiswa yang memenuhi kriteria prestasi akademik atau kondisi ekonomi tertentu sesuai syarat masing-masing beasiswa."}
    ]

    if st.button("🚀 Jalankan Evaluasi Sekarang"):
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "GANTI_DENGAN_KEY_ANDA":
            st.error("GOOGLE_API_KEY belum diatur di file .env. Dapatkan kunci API gratis di Google AI Studio dan tambahkan ke .env.")
            st.stop()
        with st.spinner("Sedang memproses evaluasi (mengumpulkan jawaban LLM)..."):
            for item in test_questions:
                # 1. Ambil konteks dari Qdrant
                docs = retriever.invoke(item["q"])
                context_texts = [doc.page_content for doc in docs]
                
                # 2. Ambil jawaban dari model lokal
                answer = rag_chain.invoke(item["q"])
                
                questions.append(item["q"])
                answers.append(answer)
                contexts.append(context_texts)
                ground_truths.append(item["gt"])

        with st.spinner("Sedang menghitung skor menggunakan Gemini Cloud..."):
            try:
                eval_df = run_ragas_evaluation(questions, answers, contexts, ground_truths)
                
                st.success("Evaluasi Selesai!")

                # ── Ringkasan metrik ──────────────────────────────────
                st.subheader("📈 Rata-rata Skor")
                metrics_to_show = ["faithfulness", "answer_relevancy", "context_quality"]
                metric_labels = {
                    "faithfulness":     "Faithfulness (Anti-Halusinasi)",
                    "answer_relevancy": "Answer Relevancy",
                    "context_quality":  "Context Quality",
                }
                cols = st.columns(len(metrics_to_show))
                for col, metric in zip(cols, metrics_to_show):
                    if metric in eval_df.columns:
                        avg = eval_df[metric].mean()
                        col.metric(
                            label=metric_labels.get(metric, metric),
                            value=f"{avg:.2f}" if not pd.isna(avg) else "N/A",
                        )

                # ── Detail per pertanyaan ─────────────────────────────
                st.subheader("🔍 Detail Hasil per Pertanyaan")
                display_cols = ["no", "question", "faithfulness", "answer_relevancy",
                                "context_quality", "alasan_faithfulness",
                                "alasan_relevancy", "alasan_context"]
                st.dataframe(eval_df[[c for c in display_cols if c in eval_df.columns]],
                             use_container_width=True)

                # ── Grafik ────────────────────────────────────────────
                st.subheader("📊 Visualisasi Metrik per Pertanyaan")
                chart_df = eval_df[["no"] + [m for m in metrics_to_show if m in eval_df.columns]].set_index("no")
                st.bar_chart(chart_df)

            except Exception as e:
                st.error(f"Gagal menjalankan evaluasi: {e}")
                st.info("Pastikan GOOGLE_API_KEY sudah benar di file .env")
