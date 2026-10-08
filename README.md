# Program Analisis Efek Program Promosi (Bab 4 Tesis)

Program ini menghitung seluruh angka pada Bab 4 tesis: gambaran umum data, pola pemberian promosi, ilustrasi promosi #8, hasil 30 program promosi (cara konvensional vs cara terkoreksi), efek rata-rata gabungan, dan tambahan biaya diskon.

## 0. Versi notebook (Jupyter / Google Colab)

Selain skrip `analisis_promosi.py`, tersedia `analisis_promosi.ipynb` dengan isi perhitungan yang sama, dipecah per bagian Bab 3–4 beserta penjelasannya. Buka di Jupyter Notebook, JupyterLab, VS Code, atau Google Colab, taruh folder `data` di lokasi yang sama, lalu jalankan semua sel (*Run All*). Notebook ini sudah berisi hasil lengkap dari satu kali penjalanan, sehingga tabel dan gambar langsung terlihat tanpa perlu dijalankan ulang.

## 1. Yang perlu disiapkan

1. **Python 3.9 atau lebih baru.** Unduh dari https://www.python.org/downloads/. Saat instalasi di Windows, centang **"Add Python to PATH"**.
2. **Data dunnhumby *The Complete Journey*.** Cukup tiga file berikut, taruh dalam satu folder bernama `data`:
   - `transaction_data.csv`
   - `campaign_table.csv`
   - `campaign_desc.csv`

Susunan folder:

```
program_tesis/
├── analisis_promosi.py
├── requirements.txt
├── README.md
└── data/
    ├── transaction_data.csv
    ├── campaign_table.csv
    └── campaign_desc.csv
```

## 2. Instalasi pustaka (sekali saja)

Buka **Command Prompt** (Windows) atau **Terminal** (Mac/Linux), masuk ke folder `program_tesis`, lalu jalankan:

```
pip install -r requirements.txt
```

## 3. Menjalankan program

```
python analisis_promosi.py
```

Program membaca data dari folder `data` dan menyimpan hasil ke folder `hasil`. Lama proses sekitar 15–30 menit dengan pengaturan bawaan (1.000 ulangan bootstrap), tergantung kecepatan komputer.

Pilihan tambahan:

| Pilihan | Fungsi | Contoh |
|---|---|---|
| `--data` | Lokasi folder data | `--data D:\tesis\data` |
| `--output` | Lokasi folder hasil | `--output hasil_final` |
| `--bootstrap` | Jumlah ulangan bootstrap untuk *standard error* (default 1000). Makin besar makin stabil, tetapi makin lama | `--bootstrap 500` |
| `--seed` | Angka awal acak, agar hasil sama setiap kali dijalankan (default 2026) | `--seed 2026` |

Contoh lengkap:

```
python analisis_promosi.py --data data --output hasil --seed 2026
```

## 4. Hasil yang diperoleh

Di folder `hasil`:

| File | Isi | Dipakai di tesis |
|---|---|---|
| `ringkasan_hasil.txt` | Angka-angka Bab 4.1–4.6 | Teks Bab 4 |
| `tabel_hasil_per_promosi.csv` / `.xlsx` | Hasil per program promosi | Tabel 4.2 dan Tabel 4.3 |
| `gambar/gambar_4_1` s.d. `gambar_4_5` (.png) | Diagram: pola pemberian promosi, ilustrasi promosi #8, konvensional vs setelah koreksi, efek per promosi dan gabungan, ringkasan | Gambar 4.1–4.5 |

Arti kolom pada tabel:

| Kolom | Arti |
|---|---|
| `promosi`, `tipe`, `durasi` | Nomor, tipe (A/B/C), dan lama program promosi (hari) |
| `pelanggan_penerima` | Jumlah pelanggan yang dikirimi program promosi |
| `kelompok_pembanding` | Jumlah pelanggan yang tidak menerima promosi apa pun selama periode tersebut |
| `jumlah_pembanding_efektif` | Jumlah pelanggan pembanding yang benar-benar mirip dengan penerima setelah pembobotan |
| `konvensional` | Efek menurut cara hitung konvensional ($ per pelanggan) |
| `setelah_koreksi`, `se` | Efek menurut cara hitung terkoreksi dan *standard error*-nya |
| `ik95_bawah`, `ik95_atas` | Interval kepercayaan 95% |
| `tambahan_diskon`, `se_diskon` | Tambahan diskon yang ditanggung perusahaan akibat promosi |

Promosi yang kolom `setelah_koreksi`-nya kosong (promosi #3) tidak dapat dihitung karena tidak ada pelanggan pembanding yang mirip.

## 5. Alur perhitungan di dalam program

1. **Membaca data** dan menghitung diskon perusahaan (`RETAIL_DISC` + `COUPON_MATCH_DISC`). Kupon produsen (`COUPON_DISC`) tidak dihitung sebagai biaya perusahaan.
2. **Untuk setiap program promosi:**
   - menentukan periode promosi dan periode sebelum (sama panjang);
   - menentukan pelanggan penerima dan kelompok pembanding (pelanggan yang tidak menerima promosi lain yang berjalan bersamaan);
   - menghitung enam karakteristik pelanggan sebelum promosi;
   - **cara konvensional:** selisih rata-rata belanja selama promosi;
   - **cara terkoreksi:** (1) bobot *entropy balancing*, (2) perubahan belanja sebelum–selama (DiD), (3) koreksi regresi linear;
   - *standard error* dengan bootstrap.
3. **Menggabungkan** hasil seluruh promosi dengan rata-rata pembobotan (bobot = 1/SE²).

## 6. Jika ada masalah

| Pesan | Penyebab dan solusi |
|---|---|
| `python is not recognized` | Python belum masuk PATH. Instal ulang Python dan centang "Add Python to PATH" |
| `No module named ...` | Jalankan lagi `pip install -r requirements.txt` |
| `File ... tidak ditemukan` | Periksa nama file dan pastikan berada di folder `data` |
| `MemoryError` | Tutup aplikasi lain; data transaksi membutuhkan sekitar 1 GB RAM |
