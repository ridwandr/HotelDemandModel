# Hotel Booking Demand Cancellation Prediction

## Ringkasan Proyek
Proyek ini membangun model **machine learning** untuk memprediksi **pembatalan reservasi kamar hotel** (binary classification). Target bisnisnya adalah **mengurangi cancellation rate** dan menjaga okupansi tetap stabil dengan memberi **peringatan dini** pada pemesanan berisiko tinggi.

**Model terbaik:** `XGBoost`  
**Akurasi:** **79%** · **F1-score (kelas cancel):** **74%**  
**Penanganan imbalance:** **oversampling** pada data latih  
**Encoding:** **Binary Encoding** untuk fitur dengan kardinalitas tinggi, **One-Hot Encoding (OHE)** untuk kardinalitas rendah

---

## Business Problem
- **Masalah:** Tingginya tingkat pembatalan membuat okupansi sulit diprediksi dan pendapatan potensial hilang.
- **Tujuan:** Mengidentifikasi pemesanan berisiko dibatalkan sejak awal untuk tindakan preventif (mis. kebijakan deposit/non-refundable, strategi overbooking yang aman).
- **Output model:** Prediksi `is_canceled` (0 = tidak, 1 = ya) lengkap dengan probabilitas untuk pemeringkatan risiko.

---

## Dataset
- **Sumber:** Sistem reservasi hotel (Portugal), data historis pemesanan yang telah dianonimkan.
- **Unit observasi:** 1 baris = 1 pemesanan.
- **Contoh fitur penting:**
  - `deposit_type`, `market_segment`, `customer_type`
  - `previous_cancellations`, `booking_changes`, `days_in_waiting_list`
  - `required_car_parking_space`, `total_of_special_request`
  - **Label:** `is_canceled`

> Catatan: Fitur dengan kardinalitas tinggi (mis. `country`, `reserved_room_type`) diperlakukan dengan **Binary Encoding**.

---

## Metodologi
1. **Data Splitting**
   - Train/Validation/Test (stratified) untuk menjaga distribusi kelas.
2. **Data Cleaning**
   - Menangani nilai hilang (imputasi sederhana: mode/median/0 sesuai konteks).
   - Cek duplikasi & *type casting* seperlunya.
3. **Feature Engineering**
   - **Encoding kategorikal:**
     - **Binary Encoding** untuk fitur *high cardinality* (contoh: `country`, `reserved_room_type`).
     - **One-Hot Encoding (OHE)** untuk fitur *low cardinality*.
   - Opsi penanganan outlier pada fitur numerik (clipping/log-transform) bila diperlukan.
4. **Imbalance Handling**
   - **Oversampling** pada data latih untuk menyeimbangkan kelas (menghindari *data leakage* dengan hanya dilakukan pada fold train).
5. **Modeling**
   - Kandidat model diuji, dipilih **XGBoost** sebagai final model berdasarkan metrik validasi.
6. **Evaluation**
   - Metrik: **Accuracy**, **Precision**, **Recall**, **F1-score** (fokus pada kelas cancel).
   - Validasi silang untuk mengurangi *variance* penilaian.
7. **Interpretasi & Monitoring**
   - Analisis fitur penting (feature importance) dan evaluasi berkala untuk *model drift*.

---

## Hasil Utama
- **Model:** XGBoost
- **Akurasi:** 79%
- **F1-score (kelas cancel):** 74%
- **Observasi:**
  - Fitur seperti **`deposit_type`**, **`market_segment`**, dan **`previous_cancellations`** konsisten berkontribusi kuat.
  - **Oversampling** meningkatkan *recall* kelas cancel tanpa menurunkan *precision* secara drastis.
  - Kombinasi **Binary Encoding + OHE** efektif menjaga dimensi fitur tetap efisien namun informatif.

---

## Dampak Bisnis yang Diusulkan
- **Risk flagging:** Beri skor risiko di sistem reservasi; tandai pemesanan *high risk*.
- **Kebijakan adaptif:** Minta deposit / *non-refundable* untuk *high risk*; longgarkan untuk *low risk*.
- **Strategi pemasaran:** Fokuskan promosi ke segmen dengan risiko rendah; lakukan intervensi (reminder, penawaran) untuk risiko menengah.
- **Operasional:** Gunakan skor risiko untuk **overbooking** yang lebih aman dan perencanaan *staffing*.

---

## Cara Menjalankan
### 1) Persiapan
```bash
git clone <repo-anda>
cd <repo-anda>
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Training Ulang (opsional)
```bash
python src/train.py   --data_path data/data_hotel_booking_demand.csv   --model_out models/xgboost_model.pkl
```

### 3) Inference Cepat (contoh)
```python
# src/inference.py (contoh penggunaan)
import pickle
import pandas as pd

with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

sample = pd.DataFrame([{
    "deposit_type": "No Deposit",
    "market_segment": "Online TA",
    "customer_type": "Transient",
    "previous_cancellations": 0,
    "booking_changes": 1,
    "days_in_waiting_list": 0,
    "required_car_parking_space": 0,
    "total_of_special_request": 1,
    "country": "PRT",
    "reserved_room_type": "A"
}])

# Pastikan preprocessing (encoding) yang sama digunakan seperti saat training
y_proba = model.predict_proba(sample)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
print(float(y_proba[0]), int(y_pred[0]))
```

---

## Catatan Implementasi Teknis
- **Encoding**
  - *High cardinality:* gunakan **Binary Encoder** (library `category_encoders`).
  - *Low cardinality:* gunakan **One-Hot Encoder** (scikit-learn).
- **Pipeline**
  - Gunakan `ColumnTransformer` + `Pipeline` agar preprocessing identik di train & inference.
- **Imbalance**
  - Lakukan **oversampling di dalam pipeline** (mis. `imblearn`’s `Pipeline`) dan **hanya pada data latih**.
- **Evaluasi**
  - Laporkan *confusion matrix* dan *classification report* selain metrik agregat.
- **Reproducibility**
  - Set `random_state` pada semua komponen yang bersifat acak.

---

## Deliverables
- **Notebook** dokumentasi proses end-to-end.
- **Model** ter-serialize (`Hotel_Booking_xgb1300_over.joblib`).
- **README.md** (dokumen ini).
- (**Opsional**) **Slide** presentasi & **video** penjelasan (≤ 10 menit).

---

## Rekomendasi Lanjutan
1. Tambah fitur operasional (mis. *lead time*, harga kamar, pola musiman) untuk akurasi lebih tinggi.
2. Lakukan *threshold tuning* berbasis biaya (cost-sensitive) agar kebijakan bisnis lebih presisi.
3. Jadwalkan *retraining* berkala untuk mengantisipasi perubahan tren.
4. Bangun *monitoring dashboard* untuk memantau drift dan performa produksi.

---

## Dependensi (contoh)
```
python==3.10
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
category-encoders
matplotlib
```

---

## 👤 Kontak
Jika ada pertanyaan atau ingin menyesuaikan struktur repo, silakan hubungi *maintainer* proyek ini.
