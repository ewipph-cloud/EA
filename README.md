# Riset Scalping XAUUSD

Repositori ini difokuskan pada eksperimen terbaru: entry M1 dengan konteks
tren M5. Ini masih prototipe riset, bukan EA siap live atau strategi yang
terbukti profit.

## Metode yang diuji

- Konteks M5 memakai EMA(5), EMA(13), kemiringan EMA(13), dan garis tengah
  Bollinger 20-bar.
- Entry M1 mengikuti arah konteks M5 saat candle M1 searah dan close melewati
  close candle sebelumnya.
- Satu posisi terbuka pada satu waktu; SL awal 0,75 ATR, trailing mulai pada
  0,25 ATR dengan jarak 0,50 ATR, tanpa TP, dan batas tahan 5 bar M1.

Laporan hasil terbaru ada di `xauusd_m1_entries_m5_context_relaxed.json`.
Rata-rata posisi per hari aktif pada train/validation/periode diagnostik adalah
sekitar 259/273/277. Target rata-rata 100 posisi tercapai, tetapi tidak pada
setiap hari. Hasil validation dan periode diagnostik berubah negatif pada
asumsi biaya 0,05R per posisi; biaya ini hipotetis, bukan spread broker.

Data historis yang tersedia hanya bid, sehingga spread, komisi, swap, dan
slippage aktual tidak diketahui. Periode diagnostik juga bukan holdout yang
belum pernah diperiksa. Karena itu hasil ini belum menunjukkan profitabilitas
untuk trading live. EA MQL5 belum dibuat.

## Menjalankan riset

Data harga tidak disertakan. Letakkan file sumber M5
`XAUUSD_M5_HistData_2022-2026.csv` dan arsip M1
`HISTDATA_XAUUSD_M1_*.zip` di direktori ini, lalu jalankan:

```bash
python clean_xauusd_m5_history.py
python research_m1_entries_m5_context.py
```

Skrip menggunakan Python standard library. File data dan hasil antara yang
besar diabaikan Git melalui `.gitignore`.
