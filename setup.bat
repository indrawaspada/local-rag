@echo off
echo Mengatur Virtual Environment Python...
python -m venv venv
call venv\Scripts\activate.bat

echo Menginstall kebutuhan (libraries)...
pip install -r requirements.txt

echo.
echo Setup Selesai!
echo Untuk memasukkan dokumen PDF ke Qdrant, jalankan:
echo venv\Scripts\activate.bat ^& python ingest.py
echo.
echo Untuk menjalankan antarmuka Web RAG, jalankan:
echo venv\Scripts\activate.bat ^& streamlit run app.py
pause
