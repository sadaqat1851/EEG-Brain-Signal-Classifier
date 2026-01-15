# 🧠 EEG Brain Signal Classifier
### Data Science × Neuroscience | Beginner-Friendly Undergraduate Project

> **Classifying mental states (Relaxed vs. Focused) from simulated EEG brainwave signals using Python and Machine Learning.**

---

## 📌 What This Project Does

This project simulates EEG (Electroencephalogram) brainwave signals, extracts meaningful numerical features, and trains a **Random Forest** machine learning model to automatically classify whether a person is in a **Relaxed** or **Focused** mental state.

No external dataset download required — the EEG data is synthetically generated using known neuroscience principles.

---

## 🔬 The Biological Background

The human brain produces tiny electrical signals that can be recorded from the scalp. These are grouped into **frequency bands**, each linked to a different mental state:

| Band | Frequency Range | Mental State |
|------|----------------|--------------|
| **δ Delta** | 0.5 – 4 Hz | Deep sleep |
| **θ Theta** | 4 – 8 Hz | Drowsy, meditative |
| **α Alpha** | 8 – 12 Hz | **Relaxed, calm** ← we simulate this |
| **β Beta** | 12 – 30 Hz | **Active, focused** ← we simulate this |
| **γ Gamma** | 30 – 100 Hz | High cognition |

Our simulated **Relaxed EEG** has dominant Alpha waves (~10 Hz).  
Our simulated **Focused EEG** has dominant Beta waves (~20 Hz).

---

## 📊 The Data Science Pipeline

```
Raw EEG Signal
      │
      ▼
 [Signal Simulation]   ← Generates realistic brainwaves using sine functions + noise
      │
      ▼
 [Feature Extraction]  ← Extracts 13 numerical features per signal epoch
      │                    (Time-domain stats + FFT frequency band powers)
      ▼
 [Dataset Building]    ← 300 labeled samples (150 Relaxed, 150 Focused)
      │
      ▼
 [Train/Test Split]    ← 80% training, 20% testing (honest evaluation)
      │
      ▼
 [Random Forest Model] ← 100 decision trees, majority vote
      │
      ▼
 [Evaluation]          ← Accuracy, Precision, Recall, Confusion Matrix
```

---

## 🧩 Features Extracted from Each EEG Epoch

### Time-Domain Features (Statistical shape of the signal)
| Feature | What It Captures |
|---------|-----------------|
| `mean` | Average voltage level |
| `std` | How much the signal varies |
| `variance` | Spread of the voltage values |
| `min` / `max` / `range` | Extreme voltage excursions |
| `skewness` | Asymmetry in the distribution |
| `kurtosis` | Presence of sharp voltage spikes (artifacts) |

### Frequency-Domain Features (FFT Power per brain band)
| Feature | What It Captures |
|---------|-----------------|
| `delta_power` | Power in 0.5–4 Hz range |
| `theta_power` | Power in 4–8 Hz range |
| `alpha_power` | Power in 8–12 Hz range ← key for Relaxed |
| `beta_power` | Power in 12–30 Hz range ← key for Focused |
| `gamma_power` | Power in 30–100 Hz range |

---

## 📁 Project Structure

```
EEG-BrainSignal-Classifier/
│
├── eeg_classifier.py        ← Main Python script (heavily commented)
├── requirements.txt         ← Python package dependencies
├── README.md                ← This file
│
└── (generated outputs)
    ├── eeg_signal_analysis.png   ← Raw EEG plots + FFT spectra
    └── model_results.png         ← Confusion matrix + feature importances
```

---

## 🚀 How to Run This Project

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/EEG-BrainSignal-Classifier.git
cd EEG-BrainSignal-Classifier
```

### Step 2 — Set up a Python virtual environment
```bash
# Create the environment
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the project
```bash
python eeg_classifier.py
```

You will see:
- Two `.png` plot files generated in the same folder
- Accuracy and classification report printed in the terminal

---

## 📈 Expected Output (Sample)

```
============================================================
  EEG Brain Signal Classifier — Beginner Friendly
  Data Science × Neuroscience
============================================================

[Step 1] Plotting EEG signals and frequency spectra...
[Step 2] Building the dataset...
Generating 150 RELAXED samples...
Generating 150 FOCUSED samples...
Dataset built! Shape: (300, 14)

[Step 4] Training and evaluating the model...
--- Training Random Forest Classifier ---

Test Accuracy: ~95%+

Detailed Classification Report:
              precision    recall  f1-score
    Relaxed       0.97      0.97      0.97
    Focused       0.97      0.97      0.97
```

---

## 🛠️ Tools & Libraries Used

| Library | Role |
|---------|------|
| `numpy` | Numerical computation, signal math |
| `scipy` | FFT (frequency analysis) |
| `pandas` | Dataset creation & manipulation |
| `scikit-learn` | Machine learning (Random Forest, evaluation) |
| `matplotlib` | Visualization & plots |

---

## 💡 Beginner Learning Goals

After completing this project, you will understand:

- ✅ What EEG signals are and why they differ between mental states
- ✅ How **FFT (Fast Fourier Transform)** decomposes a signal into frequencies
- ✅ How to extract features from biological time-series data
- ✅ How to build and evaluate a **Random Forest** classifier
- ✅ What a **confusion matrix** and **classification report** tell you

---

## 🔗 Acknowledgements & References

This project was independently built as a beginner-friendly introduction to the intersection of neuroscience and machine learning.

Key references:
- [MNE-Python EEG Library](https://mne.tools) — for real EEG data in future
- [PhysioNet EEG Datasets](https://physionet.org) — for real data
- [Scikit-learn Documentation](https://scikit-learn.org)
- [SciPy FFT Documentation](https://docs.scipy.org/doc/scipy/reference/fft.html)

---

## 📬 Author

**Your Name**  
Undergraduate Student | Data Science + Bioscience  
GitHub: [@your-username](https://github.com/your-username)

---

*Built as part of an undergraduate exploration into the intersection of data science and neuroscience.*
