# Tugas Besar 1 IF3270 Pembelajaran Mesin
## Feedforward Neural Network (FFNN) from Scratch

Implementasi Feedforward Neural Network (FFNN) dari nol menggunakan Python dan NumPy, untuk memenuhi spesifikasi Tugas Besar 1 IF3270 Pembelajaran Mesin 2025/2026.

---

## Struktur Repository

```
├── README.md
├── data/
│   └── global_student_placement_and_salary.csv   # Dataset (taruh di sini)
├── doc/
│   └── laporan.pdf                                # Laporan tugas
└── src/
```

---

## Setup & Cara Menjalankan

### 1. Clone repository

```bash
git clone <url-repository>
cd Tubes
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan dataset

Download dataset `datasetml_2026.csv` dan letakkan di folder `data/`.

### 5. Jalankan notebook

```bash
cd src
jupyter notebook
```

Buka `FFNN.ipynb` untuk melihat implementasi.

---

## Dependensi

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (untuk preprocessing dan perbandingan)
- pandas
- tqdm
- jupyter

---

## Pembagian Tugas

| Nama | NIM | Tugas |
|------|-----|-------|
| Bryan Ho | 13523029 | Membuat model.py, Membantu membuat bonus fungsi activation, Membuat bonus metode inisialisasi bobot, Membantu membuat laporan, Membantu membuat main.ipynb. |
| Alvin Christopher Santausa | 13523033 | Membuat activation.py, Membuat bonus autograd.py, Membantu membuat laporan, Membuat main.ipynb. |
| Kenneth Poenadi | 13523040 | Membuat layer.py sebagai base model untuk dipakai pada model.py, Membantu membuat loss.py, Membuat RMSNorm, Membuat Optimizer Adam, Membantu membuat laporan. |

---

## Hasil Pengujian

Dataset: `datasetml_2026.csv` — klasifikasi biner `placement_status` (Placed / Not Placed)  
Split: 80% train / 20% test (stratified) | Preprocessing: StandardScaler  
Arsitektur baseline: `Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)`

---
