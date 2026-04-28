@echo off
echo Membuat atau memperbarui konfigurasi memori WSL...
set WSL_CONFIG="%USERPROFILE%\.wslconfig"

echo [wsl2] > %WSL_CONFIG%
echo memory=10GB >> %WSL_CONFIG%
echo processors=4 >> %WSL_CONFIG%

echo Konfigurasi .wslconfig berhasil dibuat di %WSL_CONFIG% dengan batas memori 10GB.
echo Merestart layanan WSL untuk menerapkan perubahan...
wsl --shutdown

echo Selesai! Silakan jalankan Docker Desktop Anda kembali.
pause
