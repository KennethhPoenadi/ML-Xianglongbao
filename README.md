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

Download dataset `global_student_placement_and_salary.csv` dan letakkan di folder `data/`.

### 5. Jalankan notebook

```bash
cd src
jupyter notebook
```

Buka `FFNN.ipynb` untuk melihat implementasi, atau `experiment.ipynb` untuk pengujian dan analisis hyperparameter.

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
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |

---

## Referensi

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
- [Forward Propagation](https://www.jasonosajima.com/forwardprop)
- [Backpropagation](https://www.jasonosajima.com/backprop)
- [NumPy Documentation](https://numpy.org/doc/2.2/)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)