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

## Hasil Pengujian

Dataset: `datasetml_2026.csv` — klasifikasi biner `placement_status` (Placed / Not Placed)  
Split: 80% train / 20% test (stratified) | Preprocessing: StandardScaler  
Arsitektur baseline: `Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)`

---

### 1. Pengaruh Depth dan Width

**Setup:** SGD lr=0.01, epochs=50, batch=32

#### Variasi Width (depth=2 hidden layers)

| Arsitektur | Test Accuracy |
|---|:---:|
| Sempit (16×2→1) | **74.95%** |
| Sedang (64×2→1) | 74.10% |
| Lebar (256×2→1) | 72.60% |

#### Variasi Depth (width=64 per layer)

| Arsitektur | Test Accuracy |
|---|:---:|
| Dangkal (64×1→1) | **74.65%** |
| Sedang (64×2→1) | 74.05% |
| Dalam (64×3→1) | 73.10% |

**Dasar Teori:** *Universal Approximation Theorem* menyatakan bahwa satu hidden layer sudah cukup untuk merepresentasikan fungsi kontinu manapun, namun kedalaman membantu mengurangi jumlah neuron yang diperlukan. Kapasitas model diukur dari total parameter yang dapat dipelajari:

$$P = \sum_{l=1}^{L} \left(n_{l-1} \cdot n_l + n_l\right)$$

di mana $n_l$ adalah jumlah neuron di layer ke-$l$. Model dengan $P$ besar memiliki kapasitas tinggi, namun rentan overfitting jika jumlah sampel $N$ tidak jauh melebihi $P$. Bias-variance tradeoff menyatakan bahwa error total $= \text{Bias}^2 + \text{Variance} + \text{Noise}$ — menambah kapasitas menurunkan bias namun menaikkan variance.

Hasil menunjukkan bahwa menambah width maupun depth justru menurunkan akurasi secara konsisten. Model sempit (16×2) dan dangkal (64×1) memiliki akurasi tertinggi, sementara model paling besar justru paling rendah. Penyebabnya adalah overfitting — val loss pada model besar memang turun cepat di awal, namun mulai merangkak naik setelah epoch ke-10 hingga ke-15, sementara model kecil mempertahankan val loss yang lebih stabil. Pada dataset dengan jumlah fitur dan sampel terbatas, model dengan parameter lebih sedikit cenderung menggeneralisasi lebih baik karena tidak "menghafal" noise training data.

---

### 2. Pengaruh Fungsi Aktivasi

**Setup:** Arsitektur 64→64→32→1, layer ke-2 divariasikan; SGD lr=0.01, epochs=50

| Aktivasi Layer-2 | Test Accuracy |
|---|:---:|
| relu | 73.95% |
| sigmoid | **74.55%** |
| tanh | 73.90% |
| linear | 73.95% |

**Dasar Teori:** Fungsi aktivasi memperkenalkan non-linieritas ke jaringan. Berikut definisi dan turunannya yang relevan untuk backpropagation:

| Fungsi | Definisi | Turunan |
|---|---|---|
| ReLU | $f(x) = \max(0, x)$ | $f'(x) = \mathbf{1}[x > 0]$ |
| Sigmoid | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $\sigma(x)\bigl(1 - \sigma(x)\bigr) \in (0, 0.25]$ |
| Tanh | $\tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x) \in (0, 1]$ |
| Linear | $f(x) = x$ | $f'(x) = 1$ |

Masalah **vanishing gradient** terjadi karena turunan sigmoid dan tanh mendekati 0 saat $|x|$ besar. Dalam backpropagation, gradient layer $l$ dikalikan berulang kali dengan turunan ini sehingga sinyal meredam eksponensial: $\delta^{(l)} = \delta^{(L)} \prod_{k=l}^{L} \sigma'(z^{(k)}) W^{(k)} \to 0$.

Perbedaan akurasi antar seluruh variasi aktivasi sangat kecil (rentang 73.90%–74.55%), menunjukkan bahwa pilihan fungsi aktivasi pada satu layer tengah bukan faktor penentu utama di dataset ini. Dari sisi kurva loss, `sigmoid` konvergen paling lambat karena masalah vanishing gradient pada nilai jenuh, namun justru menghasilkan val loss paling rendah di akhir training. `relu`, `tanh`, dan `linear` memiliki kecepatan konvergensi sebanding dengan kurva loss yang hampir identik. Kesimpulannya, selama fungsi aktivasi yang digunakan bukan softmax (yang tidak cocok untuk hidden layer biner), perbedaannya tidak signifikan secara praktis — faktor arsitektur dan learning rate jauh lebih berpengaruh.

---

### 3. Pengaruh Learning Rate

**Setup:** Arsitektur 64→32→1, SGD, epochs=50, batch=32

| Learning Rate | Test Accuracy |
|---|:---:|
| 0.001 (sangat kecil) | 73.95% |
| 0.01 (default) | **74.50%** |
| 0.1 (terlalu besar) | 70.35% |

**Dasar Teori:** Pada Mini-batch Gradient Descent, parameter diperbarui setiap batch dengan:

$$\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} \mathcal{L}(\theta_t)$$

di mana $\alpha$ adalah learning rate. Pemilihan $\alpha$ kritis karena:

- **$\alpha$ terlalu kecil:** langkah update kecil, loss turun sangat lambat — setelah 50 epoch model belum mendekati optimum.
- **$\alpha$ terlalu besar:** update melampaui minimum, bobot berosilasi. Untuk fungsi loss dengan konstanta Lipschitz $L$, konvergensi terjamin hanya jika $\alpha < \frac{2}{L}$. Jika tidak, update memperbesar loss alih-alih mengecilkannya.

`lr=0.01` adalah sweet spot — train dan val loss konvergen stabil. `lr=0.001` masih dalam fase penurunan di epoch ke-50 (train loss $\approx 0.51$, belum plateau). `lr=0.1` mulai divergen sekitar epoch ke-15: gradient update terlalu besar sehingga bobot melompat jauh dari basin optimum, menyebabkan val loss membesar hingga $\approx 0.73$ di epoch ke-50.

---

### 4. Pengaruh Inisialisasi Bobot

**Setup:** Arsitektur 64→32→1, Adam lr=0.001, epochs=100, batch=32

| Metode | Train Loss | Val Loss | Test Accuracy |
|---|:---:|:---:|:---:|
| zeros | 0.6664 | 0.6662 | 61.55% |
| uniform | 0.3034 | 0.7625 | **69.65%** |
| normal | 0.3271 | 0.7866 | 70.05% |
| xavier | 0.2361 | 0.8847 | 69.90% |
| he | 0.2179 | 0.9397 | 69.35% |
| auto | 0.2236 | 0.9710 | 68.30% |

**Dasar Teori:** Inisialisasi bobot mengontrol distribusi awal aktivasi dan gradient. Dua metode berbasis analisis variansi:

**Xavier / Glorot** (cocok untuk sigmoid/tanh, menjaga variansi signal maju dan gradient mundur):
$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \quad W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$$

**He / Kaiming** (cocok untuk ReLU, memperhitungkan bahwa ReLU membuang setengah aktivasi):
$$\text{Var}(W) = \frac{2}{n_{\text{in}}}, \quad W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

Inisialisasi **zeros** melanggar *symmetry breaking*: semua neuron menghasilkan output dan gradient identik, sehingga seluruh bobot selalu berubah dengan jumlah yang sama — multiple neurons kolaps menjadi ekuivalen dengan satu neuron. Hasilnya, model hanya dapat mempelajari majority class (acc $\approx$ 61.55% $\approx$ prior kelas).

`zeros` gagal total karena symmetry breaking — val loss stagnan di $\approx 0.666$ ($-\log(0.5)$, prediksi acak). `he` dan `xavier` menurunkan train loss sangat agresif ($\approx 0.22$) karena variansi bobot lebih besar memudahkan aktivasi keluar dari zona linear, namun val loss terus naik hingga 0.94 dan 0.88 (overfit berat). `uniform`/`normal` dengan variansi lebih kecil memperlambat penurunan train loss namun menghasilkan generalisasi lebih baik — gap train-val loss lebih kecil karena kecepatan update lebih terkendali. Eksperimen ini menegaskan bahwa tanpa early stopping, inisialisasi besar seperti He justru kontraproduktif.

---

### 5. Pengaruh Regularisasi

**Setup:** Arsitektur 64→32→1, SGD lr=0.01, epochs=50, batch=32

| Konfigurasi | Test Accuracy |
|---|:---:|
| Tanpa Regularisasi | 74.15% |
| L1 (λ=0.001) | **74.20%** |
| L2 (λ=0.001) | 73.85% |

**Dasar Teori:** Regularisasi menambahkan penalti pada fungsi loss untuk membatasi magnitude bobot:

**L1 (Lasso):**
$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \sum_{i} |w_i|, \qquad \frac{\partial \mathcal{L}_{\text{reg}}}{\partial w_i} = \frac{\partial \mathcal{L}}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

**L2 (Ridge / Weight Decay):**
$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{i} w_i^2, \qquad \frac{\partial \mathcal{L}_{\text{reg}}}{\partial w_i} = \frac{\partial \mathcal{L}}{\partial w_i} + \lambda w_i$$

L1 menghasilkan bobot *sparse* (banyak bobot tepat nol) karena penaltinya konstan terhadap tanda. L2 menyusutkan bobot proporsional (*weight decay*) tanpa pemangkasan. Efek regularisasi terasa signifikan hanya jika $\lambda \cdot \|\mathbf{w}\|$ sebanding dengan magnitudo gradient utama $\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{w}}\right\|$.

Ketiga konfigurasi menghasilkan akurasi yang sangat mirip karena $\lambda = 0.001$ terlalu kecil — gradient penalty $\lambda |w_i|$ atau $\lambda w_i$ jauh lebih kecil dari gradient loss utama untuk model 64→32→1 ini sehingga tidak mengubah jalur optimisasi secara signifikan. Agar regularisasi berdampak nyata, diperlukan $\lambda \geq 0.01$ atau dikombinasikan dengan early stopping.

---

### 6. Pengaruh Normalisasi RMSNorm

**Setup:** Arsitektur 64→[RMSNorm]→32→[RMSNorm]→1, SGD lr=0.01, epochs=100, batch=32

| Model | Test Accuracy | Best Epoch | Min Val Loss | Final Val Loss |
|---|:---:|:---:|:---:|:---:|
| Tanpa Normalisasi | 73.35% | 23 | 0.5120 | 0.5394 |
| Dengan RMSNorm | **73.45%** | 26 | 0.5176 | 0.5581 |

**Dasar Teori:** RMSNorm menormalisasi aktivasi menggunakan *Root Mean Square* tanpa mengurangi mean (berbeda dari LayerNorm):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \boldsymbol{\gamma}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$$

di mana $\boldsymbol{\gamma}$ adalah *learnable gain* (vektor parameter yang dapat dipelajari). Normalisasi mengurangi *internal covariate shift* — distribusi input ke tiap layer dijaga lebih stabil selama training, sehingga gradient mengalir lebih merata dan learning rate efektif lebih konsisten antar layer.

RMSNorm memberikan peningkatan marginal (+0.10%) dan best epoch lebih lambat (26 vs 23), mengindikasikan konvergensi yang lebih terkontrol berkat stabilisasi distribusi aktivasi. Namun val loss final lebih tinggi (0.5581 vs 0.5394) karena parameter $\boldsymbol{\gamma}$ yang learnable dapat memperbesar skala aktivasi secara adaptif pasca best epoch, mempercepat overfitting. RMSNorm lebih bermanfaat pada arsitektur yang lebih dalam di mana covariate shift antar layer lebih parah.

---

### 7. Perbandingan dengan sklearn MLPClassifier

**Hyperparameter identik:** hidden\_layers=(64,32), activation=relu, solver=sgd, lr=0.01, max\_iter=50, batch=32

| Model | Test Accuracy |
|---|:---:|
| FFNN Custom (buatan sendiri) | **74.55%** |
| sklearn MLPClassifier | 69.55% |
| Selisih | **+5.00%** |

> sklearn mengeluarkan `ConvergenceWarning` — 50 iterasi belum cukup konvergen.

**Dasar Teori:** Kedua model sama-sama mengoptimasi Binary Cross-Entropy:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

Namun terdapat perbedaan implementasi kritis: `sklearn.MLPClassifier` dengan `solver='sgd'` menggunakan learning rate **konstan** dari `learning_rate_init`, sedangkan internal batch iteration-nya berbasis *partial fit* yang menghitung epoch secara berbeda. Inisialisasi default sklearn adalah **Glorot uniform** terlepas dari fungsi aktivasi yang dipilih, sementara implementasi custom menggunakan He untuk ReLU. Konvergensi SGD dijamin menuju minimum lokal jika $\sum_t \alpha_t = \infty$ dan $\sum_t \alpha_t^2 < \infty$ (Robbins-Monro conditions), yang tidak terpenuhi dengan $\alpha$ konstan dan `max_iter` terbatas.

FFNN custom mengungguli sklearn MLP sebesar **+5.00%** meskipun hyperparameter identik. Penyebab utama: sklearn mengeluarkan `ConvergenceWarning` artinya bobot belum mencapai konvergensi dalam 50 iterasi — berbeda dengan implementasi custom yang menghitung 50 epoch penuh per batch secara konsisten. Dengan `max_iter` yang lebih besar (misal 200), gap ini kemungkinan akan menyempit karena keduanya konvergen ke solusi yang serupa.

---

## Referensi

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
- [Forward Propagation](https://www.jasonosajima.com/forwardprop)
- [Backpropagation](https://www.jasonosajima.com/backprop)
- [NumPy Documentation](https://numpy.org/doc/2.2/)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)