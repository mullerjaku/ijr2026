import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- KONFIGURACE ---
NUM_FILES = 20
MAX_EPOCH = 250
BASE_PATH_CUR = "/home/citic_gii/iwinac2026/results_data/cur_"
WINDOW_SIZE = 20
ENV_CHANGE_EPOCH = 150

MY_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
UM_LABELS = {
    0: 'Linear Regression',
    1: 'SVR (RBF)',
    2: 'Gradient Boosting',
    3: 'MLP Regressor'
}

def get_detailed_um_data(base_path, num_files, max_epoch):
    all_ids = set()
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[3] != 'None':
                        try:
                            all_ids.add(int(float(parts[3])))
                        except ValueError: continue
    
    sorted_ids = sorted(list(all_ids))
    results = {uid: np.zeros((num_files, max_epoch + 1)) for uid in sorted_ids}
    
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[3] != 'None':
                        try:
                            epoch = int(parts[0])
                            um_id = int(float(parts[3]))
                            if epoch <= max_epoch:
                                results[um_id][i, epoch] = 1
                        except ValueError: continue
    return results, sorted_ids

# --- NAČTENÍ DAT ---
um_data_detailed, sorted_um_ids = get_detailed_um_data(BASE_PATH_CUR, NUM_FILES, MAX_EPOCH)

# --- VYKRESLOVÁNÍ ---
plt.figure(figsize=(10, 6))
epochs = np.arange(MAX_EPOCH + 1)

for i, um_id in enumerate(sorted_um_ids):
    matrix = um_data_detailed[um_id]
    
    # Výpočty
    mean_vals = np.mean(matrix, axis=0)
    std_vals = np.std(matrix, axis=0)
    
    # Vyhlazení (Gaussian)
    mean_smoothed = pd.Series(mean_vals).rolling(window=WINDOW_SIZE, center=True, min_periods=1, win_type='gaussian').mean(std=5)
    std_smoothed = pd.Series(std_vals).rolling(window=WINDOW_SIZE, center=True, min_periods=1, win_type='gaussian').mean(std=5)
    
    # Fixace epochy 0 pro přesný začátek
    mean_smoothed.iloc[0] = mean_vals[0]
    std_smoothed.iloc[0] = std_vals[0]
    
    color = MY_COLORS[i % len(MY_COLORS)]
    label = UM_LABELS.get(um_id, f"Unknown ID: {um_id}")
    
    # Čára a stín (bez použití ax objektu)
    plt.plot(epochs, mean_smoothed, label=label, color=color, linewidth=2.5)
    
    lower_bound = np.clip(mean_smoothed - std_smoothed, 0, 1)
    upper_bound = np.clip(mean_smoothed + std_smoothed, 0, 1)
    plt.fill_between(epochs, lower_bound, upper_bound, color=color, alpha=0.12)

# Vertikální čára a popisek události
plt.axvline(x=ENV_CHANGE_EPOCH, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
plt.text(ENV_CHANGE_EPOCH + 2, 1.03, 'Goal position change', fontsize=12, color='black')

# Formátování pomocí plt
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Mean Activity Level (± SD, Gaussian smoothing, window = 20)", fontsize=12)

plt.xticks(np.arange(0, MAX_EPOCH + 1, 25))
plt.ylim(-0.02, 1.1)
plt.xlim(0, MAX_EPOCH)
plt.grid(True, linestyle='-', alpha=0.2)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), 
           ncol=4, frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig('plot_ijcr_um_activity.png', dpi=300)
plt.show()