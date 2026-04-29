import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Konfigurasi
DOCUMENTS_DIR = "./documents"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_rag_docs"
EMBEDDING_MODEL = "BAAI/bge-m3" # Menggunakan model embedding multi-bahasa terbaik
def main():
    print("Memulai proses Ingestion dokumen...")
    
    # 1. Memuat semua PDF dari folder documents
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Folder {DOCUMENTS_DIR} tidak ditemukan. Membuat folder baru...")
        os.makedirs(DOCUMENTS_DIR)
        
    print(f"Membaca dokumen PDF dari folder {DOCUMENTS_DIR}...")
    loader = DirectoryLoader(DOCUMENTS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Tidak ada dokumen PDF yang ditemukan. Harap masukkan file PDF ke dalam folder 'documents'.")
        return
        
    print(f"Ditemukan {len(documents)} halaman/bagian dokumen.")

    # 2. Memotong teks menjadi bagian kecil (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Dokumen dipotong menjadi {len(chunks)} chunks.")

    # 3. Membuat embeddings menggunakan HuggingFace (bge-m3)
    print(f"Menggunakan model embedding: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, # Ganti ke 'cuda' jika memiliki GPU NVIDIA
        encode_kwargs={'normalize_embeddings': True} # BGE models merekomendasikan normalisasi
    )

    # 4. Menyimpan ke Qdrant
    print("Menghubungkan ke Qdrant dan menyimpan data...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Buat collection jika belum ada
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    # Masukkan data ke Qdrant
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    
    print("Selesai! Semua dokumen berhasil dimasukkan ke dalam Qdrant.")

if __name__ == "__main__":
    main()
