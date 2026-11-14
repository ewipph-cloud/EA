# Riset Scalping XAUUSD

Kumpulan skrip Python untuk riset pola swing/candle, trendline, indikator,
dan trailing-stop scalping XAUUSD. Skrip menggunakan Python standard library.

**Status:** eksperimen historis, bukan EA siap live dan belum membuktikan
profitabilitas setelah biaya broker. Hasil uji terbaru tetap negatif pada
periode diagnostik: konfigurasi terbaik yang dipilih dari data train
memperoleh rata-rata `-0.0212R` setelah simulasi biaya `0.02R` per round trip.
Biaya tersebut hanya asumsi, bukan spread broker yang terukur.

## Data

Data harga tidak disertakan. Sumber historis yang dipakai adalah feed bid-only;
spread, komisi, swap, dan slippage broker tidak tersedia. Arsip M1 memiliki
timestamp berulang/konflik yang ditangani dengan mengecualikan bar ambigu.

Untuk menjalankan riset, tempatkan file sumber data yang diperlukan di direktori
ini. Nama file yang dipakai skrip tercantum pada konstanta `DATA_PATH` atau
di bagian `HISTDATA_XAUUSD_M1_*.zip`.

## Eksperimen utama

```bash
python clean_xauusd_m5_history.py
python backtest_m5_signal_m1_trailing.py
python optimize_m1_trailing_stops.py
```

Jalankan `python <nama_skrip>.py` untuk eksperimen lain. Hasil JSON ditulis ke
direktori yang sama.

Semua hasil adalah simulasi riset. Jangan gunakan sebagai rekomendasi trading
atau mengirim order live tanpa validasi terpisah pada data broker dan akun demo.
