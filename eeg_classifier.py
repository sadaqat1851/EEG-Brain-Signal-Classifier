# =============================================================================
#  EEG Brain Signal Classifier - Beginner Friendly Version
#  Author: Your Name
#  Project: Intersection of Data Science & Neuroscience
# =============================================================================
#
#  BIOLOGICAL BACKGROUND:
#  The human brain produces tiny electrical signals that can be recorded
#  from the scalp using electrodes. These signals are called EEG (Electro-
#  EncephaloGram). The brain operates across different "frequency bands":
#
#   - Delta  (0.5 - 4 Hz)  : Deep sleep
#   - Theta  (4 - 8 Hz)    : Drowsiness, meditation
#   - Alpha  (8 - 12 Hz)   : Relaxed, calm, eyes closed
#   - Beta   (12 - 30 Hz)  : Active thinking, focused
#   - Gamma  (30 - 100 Hz) : High-level cognition
#
#  DATA SCIENCE GOAL:
#  We will simulate EEG signals for two mental states (RELAXED vs FOCUSED),
#  extract simple numerical features, and train a Machine Learning model
#  to automatically classify which state a person is in.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
#  STEP 1: SIMULATE EEG DATA
#  (In a real project, you would load a .csv or .edf file here)
# =============================================================================

def simulate_eeg_signal(state='relaxed', duration_sec=2, sampling_rate=256):
    """
    Simulates a synthetic EEG signal for a given mental state.

    BIOLOGY: Different mental states produce different dominant frequencies.
             Relaxed brains show strong Alpha (8-12 Hz) activity.
             Focused brains show strong Beta (12-30 Hz) activity.

    Parameters:
        state         : 'relaxed' or 'focused'
        duration_sec  : how many seconds of data to generate
        sampling_rate : samples per second (256 Hz is common in EEG devices)

    Returns:
        t      : time axis (array of time points)
        signal : the simulated EEG voltage values (in microvolts, µV)
    """
    # Create a time axis: e.g., for 2 seconds at 256 Hz → 512 time points
    num_samples = duration_sec * sampling_rate
    t = np.linspace(0, duration_sec, num_samples)

    if state == 'relaxed':
        # RELAXED STATE: Dominant Alpha waves (10 Hz) with some Theta (6 Hz)
        # and small background noise
        eeg = (
            15 * np.sin(2 * np.pi * 10 * t) +   # Strong Alpha  (10 Hz)
             5 * np.sin(2 * np.pi * 6 * t)  +   # Mild Theta    (6 Hz)
             2 * np.random.randn(num_samples)     # Biological noise
        )
    elif state == 'focused':
        # FOCUSED STATE: Dominant Beta waves (20 Hz) with some Alpha (9 Hz)
        # and slightly more noise (active thinking = more signal complexity)
        eeg = (
            12 * np.sin(2 * np.pi * 20 * t) +   # Strong Beta   (20 Hz)
             4 * np.sin(2 * np.pi * 9 * t)  +   # Background Alpha (9 Hz)
             4 * np.random.randn(num_samples)     # More noise when focused
        )
    else:
        raise ValueError("State must be 'relaxed' or 'focused'")

    return t, eeg


# =============================================================================
#  STEP 2: FEATURE EXTRACTION
#  Convert raw EEG waveform into a set of numbers (features) that a
#  machine learning model can understand.
# =============================================================================

def extract_features(eeg_signal, sampling_rate=256):
    """
    Extracts meaningful numerical features from a raw EEG signal.

    DATA SCIENCE: Raw waveforms can't go directly into most ML models.
                  We need to summarize the signal using descriptive statistics
                  and frequency-domain analysis (FFT).

    Features extracted:
      Time-domain (shape of the wave):
        - mean, std, variance, min, max, range, skewness, kurtosis

      Frequency-domain (which frequencies dominate):
        - Power in Delta, Theta, Alpha, Beta, Gamma bands
    """
    features = {}

    # --- TIME DOMAIN FEATURES ---
    # These describe the statistical shape of the raw voltage signal
    features['mean']     = np.mean(eeg_signal)
    features['std']      = np.std(eeg_signal)
    features['variance'] = np.var(eeg_signal)
    features['min']      = np.min(eeg_signal)
    features['max']      = np.max(eeg_signal)
    features['range']    = np.max(eeg_signal) - np.min(eeg_signal)

    # Skewness: is the distribution of voltages symmetric?
    mean = np.mean(eeg_signal)
    std  = np.std(eeg_signal)
    features['skewness'] = np.mean(((eeg_signal - mean) / std) ** 3)

    # Kurtosis: are there extreme voltage spikes (common in artifacts)?
    features['kurtosis'] = np.mean(((eeg_signal - mean) / std) ** 4)

    # --- FREQUENCY DOMAIN FEATURES (using Fast Fourier Transform - FFT) ---
    # FFT decomposes the signal into its component frequencies
    # BIOLOGY: This tells us which brainwave bands (Alpha, Beta, etc.) are active
    n = len(eeg_signal)
    freqs    = fftfreq(n, d=1/sampling_rate)  # Frequency axis in Hz
    fft_vals = np.abs(fft(eeg_signal))        # Magnitude of each frequency

    # Only look at positive frequencies (FFT is symmetric)
    pos_mask = freqs >= 0
    freqs    = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    # Define frequency band ranges (in Hz)
    bands = {
        'delta_power': (0.5, 4),
        'theta_power': (4, 8),
        'alpha_power': (8, 12),
        'beta_power' : (12, 30),
        'gamma_power': (30, 100)
    }

    # Calculate the power (sum of squared magnitudes) within each band
    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs < high)
        features[band_name] = np.sum(fft_vals[band_mask] ** 2)

    return features


# =============================================================================
#  STEP 3: BUILD THE DATASET
#  Generate many EEG signal samples and extract features from each one.
# =============================================================================

def build_dataset(n_samples_per_class=150):
    """
    Creates a labeled dataset of EEG features.
    Each row = one 2-second EEG recording with its extracted features.
    Label 0 = Relaxed,  Label 1 = Focused
    """
    all_features = []
    all_labels   = []

    print(f"Generating {n_samples_per_class} RELAXED samples...")
    for _ in range(n_samples_per_class):
        _, eeg = simulate_eeg_signal(state='relaxed')
        feats  = extract_features(eeg)
        all_features.append(feats)
        all_labels.append(0)  # 0 = Relaxed

    print(f"Generating {n_samples_per_class} FOCUSED samples...")
    for _ in range(n_samples_per_class):
        _, eeg = simulate_eeg_signal(state='focused')
        feats  = extract_features(eeg)
        all_features.append(feats)
        all_labels.append(1)  # 1 = Focused

    df        = pd.DataFrame(all_features)
    df['label'] = all_labels
    print(f"\nDataset built! Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    return df


# =============================================================================
#  STEP 4: VISUALIZE THE EEG SIGNALS
#  Always visualize your data before modelling — it's good practice!
# =============================================================================

def plot_eeg_and_fft():
    """
    Plots the raw EEG waveforms and their FFT (frequency spectra)
    for both mental states side by side.
    """
    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    fig.suptitle('EEG Brain Signal Analysis\nRelaxed vs Focused Mental States',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    colors = {'relaxed': '#00d4aa', 'focused': '#ff6b6b'}

    for i, state in enumerate(['relaxed', 'focused']):
        t, eeg = simulate_eeg_signal(state=state, duration_sec=2)
        color  = colors[state]
        label  = state.capitalize()

        # --- Raw EEG Time Series Plot ---
        ax1 = fig.add_subplot(gs[0, i])
        ax1.plot(t, eeg, color=color, linewidth=0.8, alpha=0.9)
        ax1.set_title(f'{label} State — Raw EEG Signal',
                      color='white', fontsize=12, pad=10)
        ax1.set_xlabel('Time (seconds)', color='#aaaaaa')
        ax1.set_ylabel('Amplitude (µV)', color='#aaaaaa')
        ax1.set_facecolor('#161b22')
        ax1.tick_params(colors='#aaaaaa')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#30363d')
        ax1.grid(True, alpha=0.2, linestyle='--', color='#444')

        # Add annotation about the dominant frequency
        dom_freq = "~10 Hz (Alpha)" if state == 'relaxed' else "~20 Hz (Beta)"
        ax1.annotate(f'Dominant: {dom_freq}', xy=(0.98, 0.95),
                     xycoords='axes fraction', ha='right', va='top',
                     color=color, fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                               edgecolor=color, alpha=0.8))

        # --- FFT Frequency Spectrum Plot ---
        n       = len(eeg)
        freqs   = fftfreq(n, d=1/256)
        fft_mag = np.abs(fft(eeg))
        pos     = freqs >= 0

        ax2 = fig.add_subplot(gs[1, i])
        ax2.fill_between(freqs[pos], fft_mag[pos], color=color, alpha=0.4)
        ax2.plot(freqs[pos], fft_mag[pos], color=color, linewidth=1.2)
        ax2.set_xlim(0, 50)
        ax2.set_title(f'{label} State — Frequency Spectrum (FFT)',
                      color='white', fontsize=12, pad=10)
        ax2.set_xlabel('Frequency (Hz)', color='#aaaaaa')
        ax2.set_ylabel('Magnitude', color='#aaaaaa')
        ax2.set_facecolor('#161b22')
        ax2.tick_params(colors='#aaaaaa')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#30363d')
        ax2.grid(True, alpha=0.2, linestyle='--', color='#444')

        # Shade frequency bands
        band_shading = [
            (0.5, 4,  '#a78bfa', 'δ Delta'),
            (4,   8,  '#60a5fa', 'θ Theta'),
            (8,  12,  '#34d399', 'α Alpha'),
            (12, 30,  '#f97316', 'β Beta'),
            (30, 50,  '#f43f5e', 'γ Gamma')
        ]
        for low, high, bc, bname in band_shading:
            ax2.axvspan(low, high, alpha=0.08, color=bc, label=bname)

        if i == 0:
            ax2.legend(loc='upper right', fontsize=7,
                       facecolor='#161b22', edgecolor='#30363d',
                       labelcolor='white')

    plt.savefig('eeg_signal_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("Plot saved as 'eeg_signal_analysis.png'")
    plt.show()


# =============================================================================
#  STEP 5: TRAIN & EVALUATE THE MACHINE LEARNING MODEL
# =============================================================================

def train_and_evaluate(df):
    """
    Trains a Random Forest classifier on the extracted EEG features.

    DATA SCIENCE: Random Forest builds many decision trees and combines
                  their votes — it works well even with small datasets
                  and is easy to interpret.
    """
    # Separate features (X) from labels (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split into training set (80%) and test set (20%)
    # The model never sees test data during training — honest evaluation!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features (mean=0, std=1)
    # Important for many ML models so no single feature dominates
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train the Random Forest model
    print("\n--- Training Random Forest Classifier ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on the unseen test set
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc * 100:.1f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Relaxed', 'Focused']))

    return clf, scaler, X_test, y_test, y_pred


def plot_results(clf, X_columns, y_test, y_pred):
    """
    Plots the Confusion Matrix and Feature Importances.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('Model Evaluation Results', color='white',
                 fontsize=15, fontweight='bold')

    # --- Confusion Matrix ---
    ax1 = axes[0]
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Relaxed', 'Focused'])
    disp.plot(ax=ax1, colorbar=False, cmap='RdYlGn')
    ax1.set_title('Confusion Matrix', color='white', fontsize=12)
    ax1.set_facecolor('#161b22')
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    for txt in ax1.texts:
        txt.set_color('black')
    fig.patch.set_facecolor('#0d1117')

    # --- Feature Importances ---
    ax2 = axes[1]
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({
        'feature':    X_columns,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(10)

    bars = ax2.barh(feat_df['feature'], feat_df['importance'],
                    color='#00d4aa', edgecolor='#30363d', height=0.6)
    ax2.set_title('Top 10 Most Important Features', color='white', fontsize=12)
    ax2.set_xlabel('Importance Score', color='#aaaaaa')
    ax2.set_facecolor('#161b22')
    ax2.tick_params(colors='#aaaaaa')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#30363d')
    ax2.grid(axis='x', alpha=0.2, linestyle='--', color='#444')

    # Color the brain-band features differently
    for bar, feat in zip(bars, feat_df['feature']):
        if 'power' in feat:
            bar.set_color('#a78bfa')

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("Results plot saved as 'model_results.png'")
    plt.show()


# =============================================================================
#  MAIN — Run Everything in Sequence
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  EEG Brain Signal Classifier — Beginner Friendly")
    print("  Data Science × Neuroscience")
    print("=" * 60)

    # Step 1: Visualize what EEG signals look like
    print("\n[Step 1] Plotting EEG signals and frequency spectra...")
    plot_eeg_and_fft()

    # Step 2: Build a labeled dataset from simulated EEG recordings
    print("\n[Step 2] Building the dataset...")
    df = build_dataset(n_samples_per_class=150)

    # Step 3: Show a preview of the dataset
    print("[Step 3] Dataset preview (first 5 rows):")
    print(df.head().to_string())

    # Step 4: Train the ML model and evaluate it
    print("\n[Step 4] Training and evaluating the model...")
    clf, scaler, X_test, y_test, y_pred = train_and_evaluate(df)

    # Step 5: Plot the evaluation results
    print("\n[Step 5] Plotting model evaluation results...")
    feature_names = df.drop('label', axis=1).columns
    plot_results(clf, feature_names, y_test, y_pred)

    print("\n" + "=" * 60)
    print("  Project Complete!")
    print("  Check the generated .png files for your report/README.")
    print("=" * 60)
