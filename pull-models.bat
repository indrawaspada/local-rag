@echo off
echo Menunggu 5 detik untuk memastikan Ollama sudah siap...
timeout /t 5 /nobreak > NUL

echo ========================================================
echo MENGUNDUH MODEL QWEN2.5:1.5B (Ini mungkin memakan waktu)
echo ========================================================
docker exec -it local-rag-ollama ollama pull qwen2.5:1.5b

echo.
echo ========================================================
echo MENGUNDUH MODEL EMBEDDING (Nomic-embed-text)
echo ========================================================
docker exec -it local-rag-ollama ollama pull nomic-embed-text

echo.
echo ========================================================
echo SELESAI! Model siap digunakan di n8n.
echo ========================================================
pause
