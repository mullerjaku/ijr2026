import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

NUM_FILES = 20
MAX_EPOCH = 250
METHODS_ORDER = ['exploration', 'exploration_path', 'improvement']

BASE_PATH_CUR = "/home/citic_gii/iwinac2026/results_data/cur_wm_"
BASE_PATH_NOV = "/home/citic_gii/iwinac2026/results_data/cur_comb_"

# Barvy a české popisky
COLORS = {'exploration': '#1f77b4', 'exploration_path': '#ff7f0e', 'improvement': '#2ca02c'}
LABELS = {'exploration': 'Discovery of new goals', 'exploration_path': 'Discover new paths for existing goals', 'improvement': 'Selected UM function for existing goals'}

def get_detailed_data(base_path, num_files, max_epoch):
    """
    Načte data tak, aby zůstala zachována informace o jednotlivých bězích.
    Vrací slovník: {metoda: 2D numpy array (soubory x epochy)}
    """
    # Inicializace matic nulami (řádky = soubory, sloupce = epochy)
    results = {m: np.zeros((num_files, max_epoch + 1)) for m in METHODS_ORDER}
    
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        epoch = int(parts[0])
                        method = parts[1]
                        if epoch <= max_epoch and method in METHODS_ORDER:
                            results[method][i, epoch] = 1
    return results

def plot_with_std(ax, detailed_data, title):
    epochs = np.arange(MAX_EPOCH + 1)
    
    for method in METHODS_ORDER:
        # Data pro danou metodu (matice soubory x epochy)
        matrix = detailed_data[method]
        
        # 1. Výpočet průměru a směrodatné odchylky přes všechny soubory
        mean_vals = np.mean(matrix, axis=0)
        std_vals = np.std(matrix, axis=0)
        
        # 2. Vyhlazení (Smoothing) pomocí Pandas
        mean_smoothed = pd.Series(mean_vals).rolling(window=20, center=True, min_periods=1, win_type='gaussian').mean(std=5)
        std_smoothed = pd.Series(std_vals).rolling(window=20, center=True, min_periods=1, win_type='gaussian').mean(std=5)
        
        # 3. Vykreslení hlavní čáry
        ax.plot(epochs, mean_smoothed, label=LABELS[method], color=COLORS[method], linewidth=2)
        
        # 4. VÝPOČET HRANIC STÍNU S OŘEZÁNÍM (CLIPPING)
        lower_bound = np.clip(mean_smoothed - std_smoothed, 0, 1)
        upper_bound = np.clip(mean_smoothed + std_smoothed, 0, 1)
        # 4. Vykreslení stínovaného rozptylu (Mean +- STD)
        # alpha určuje průhlednost stínu
        ax.fill_between(epochs, 
                        lower_bound, 
                        upper_bound, 
                        color=COLORS[method], 
                        alpha=0.15)

    ax.set_title(title, fontsize=16, fontweight='bold', loc='left', pad=15)
    #ax.set_ylabel("Mean Activity Level", fontsize=11)
    ax.set_xticks(np.arange(0, MAX_EPOCH + 1, 25))
    ax.set_ylim(-0.1, 1.1) # Normalizováno na 0-1 (procento běhů)
    ax.grid(True, linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Popisek nad čárou (v horní části grafu)
    ax1.text(150 + 2, 1.03, 'World Model change', 
            fontsize=12, color='black')
    ax2.text(150 + 2, 1.03, 'WM & Goal position change', 
            fontsize=12, color='black')

# --- HLAVNÍ PROCES ---

# Načtení dat (předpokládá existenci BASE_PATH_CUR/NOV z předchozího kódu)
data_cur_detailed = get_detailed_data(BASE_PATH_CUR, NUM_FILES, MAX_EPOCH)
data_nov_detailed = get_detailed_data(BASE_PATH_NOV, NUM_FILES, MAX_EPOCH)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

plot_with_std(ax1, data_cur_detailed, "Motivational Engine - WM change")
plot_with_std(ax2, data_nov_detailed, "Motivational Engine - Combination WM & Goal change")

ax2.set_xlabel("Epochs", fontsize=12)
fig.supylabel("Mean Activity Level (± SD, Gaussian smoothing, window = 20)", 
               fontsize=12, x=0.07)

# Společná legenda pro oba grafy
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=12)

#plt.tight_layout()
plt.savefig('plot_ijcr_activity_plot_wm.png', dpi=300)
plt.show()