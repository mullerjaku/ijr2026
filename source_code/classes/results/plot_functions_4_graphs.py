import matplotlib.pyplot as plt
import numpy as np

num_files_cur = 10
base_path_cur = "/home/citic_gii/iwinac2026/results_data/cur_"
base_path_0 = "/home/citic_gii/iwinac2026/results_data/cur_lin_"
base_path_1 = "/home/citic_gii/iwinac2026/results_data/cur_svr_"
base_path_2 = "/home/citic_gii/iwinac2026/results_data/cur_grad_"
base_path_3 = "/home/citic_gii/iwinac2026/results_data/cur_mlp_"
steps_data_cur = {}
steps_data_0 = {}
steps_data_1 = {}
steps_data_2 = {}
steps_data_3 = {}


for i in range(num_files_cur):
    filename = f"{base_path_cur}{i}.txt"
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    epoch = int(parts[0])
                    step = int(parts[2])

                    # if epoch >= 250:
                    #     continue
                    
                    # Pokud epochu vidíme poprvé, vytvoříme pro ni seznam
                    if epoch not in steps_data_cur:
                        steps_data_cur[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_cur[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_cur_avg = []
steps_cur_avg = []
steps_cur_p25 = []
steps_cur_p75 = []

# Projdeme seřazené epochy a vypočítáme průměr
for epoch in sorted(steps_data_cur.keys()):
    steps_for_this_epoch = steps_data_cur[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_cur_avg.append(epoch)
    steps_cur_avg.append(median_step)
    steps_cur_p25.append(p25)
    steps_cur_p75.append(p75)

print("Průměrování 'vystup_cur' dokončeno.")

# --- Část 2: Načtení dat pro "vystup_nov" (beze změny) ---


for i in range(num_files_cur):
    filename = f"{base_path_0}{i}.txt"
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    epoch = int(parts[0])
                    step = int(parts[2])

                    # if epoch >= 250:
                    #     continue
                    
                    # Pokud epochu vidíme poprvé, vytvoříme pro ni seznam
                    if epoch not in steps_data_0:
                        steps_data_0[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_0[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_0_avg = []
steps_0_avg = []
steps_0_p25 = []
steps_0_p75 = []


for epoch in sorted(steps_data_0.keys()):
    steps_for_this_epoch = steps_data_0[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_0_avg.append(epoch)
    steps_0_avg.append(median_step)
    steps_0_p25.append(p25)
    steps_0_p75.append(p75)

print("Průměrování 'vystup_0' dokončeno.")

# --- Část 2: Načtení dat pro "vystup_nov" (beze změny) ---

for i in range(num_files_cur):
    filename = f"{base_path_1}{i}.txt"
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    epoch = int(parts[0])
                    step = int(parts[2])

                    # if epoch >= 250:
                    #     continue
                    
                    # Pokud epochu vidíme poprvé, vytvoříme pro ni seznam
                    if epoch not in steps_data_1:
                        steps_data_1[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_1[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_1_avg = []
steps_1_avg = []
steps_1_p25 = []
steps_1_p75 = []


for epoch in sorted(steps_data_1.keys()):
    steps_for_this_epoch = steps_data_1[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_1_avg.append(epoch)
    steps_1_avg.append(median_step)
    steps_1_p25.append(p25)
    steps_1_p75.append(p75)

print("Průměrování 'vystup_1' dokončeno.")

# --- Část 2: Načtení dat pro "vystup_nov" (beze změny) ---

for i in range(num_files_cur):
    filename = f"{base_path_2}{i}.txt"
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    epoch = int(parts[0])
                    step = int(parts[2])

                    # if epoch >= 250:
                    #     continue
                    
                    # Pokud epochu vidíme poprvé, vytvoříme pro ni seznam
                    if epoch not in steps_data_2:
                        steps_data_2[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_2[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_2_avg = []
steps_2_avg = []
steps_2_p25 = []
steps_2_p75 = []


for epoch in sorted(steps_data_2.keys()):
    steps_for_this_epoch = steps_data_2[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_2_avg.append(epoch)
    steps_2_avg.append(median_step)
    steps_2_p25.append(p25)
    steps_2_p75.append(p75)

print("Průměrování 'vystup_2' dokončeno.")

# --- Část 2: Načtení dat pro "vystup_nov" (beze změny) ---

for i in range(num_files_cur):
    filename = f"{base_path_3}{i}.txt"
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    epoch = int(parts[0])
                    step = int(parts[2])

                    # if epoch >= 250:
                    #     continue
                    
                    # Pokud epochu vidíme poprvé, vytvoříme pro ni seznam
                    if epoch not in steps_data_3:
                        steps_data_3[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_3[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_3_avg = []
steps_3_avg = []
steps_3_p25 = []
steps_3_p75 = []


for epoch in sorted(steps_data_3.keys()):
    steps_for_this_epoch = steps_data_3[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_3_avg.append(epoch)
    steps_3_avg.append(median_step)
    steps_3_p25.append(p25)
    steps_3_p75.append(p75)

print("Průměrování 'vystup_3' dokončeno.")

# --- Plotting in a 2x2 Grid ---

# Configuration for the subplots: (Label, Data_Epochs, Data_Avg, Data_P25, Data_P75, Color)
comparisons = [
    ('LNR', epochs_0_avg, steps_0_avg, steps_0_p25, steps_0_p75, 'green'),
    ('SVR', epochs_1_avg, steps_1_avg, steps_1_p25, steps_1_p75, 'orange'),
    ('GBR', epochs_2_avg, steps_2_avg, steps_2_p25, steps_2_p75, 'brown'),
    ('MLP', epochs_3_avg, steps_3_avg, steps_3_p25, steps_3_p75, 'grey')
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten() # Flatten 2x2 array into a list of 4 for easy looping

for i, (label, ep, avg, p25, p75, color) in enumerate(comparisons):
    ax = axes[i]
    
    # 1. Plot "Motivational Engine" (The Constant Reference)
    ax.plot(epochs_cur_avg, steps_cur_avg, color='blue', label='Motivational Engine', linewidth=2)
    ax.fill_between(epochs_cur_avg, steps_cur_p25, steps_cur_p75, color='blue', alpha=0.1)

    # 2. Plot the specific algorithm for this subplot
    ax.plot(ep, avg, color=color, label=label, linewidth=2)
    ax.fill_between(ep, p25, p75, color=color, alpha=0.2)

    # 3. Aesthetics and Labels
    ax.set_title(f"Motivational Engine vs {label}", fontsize=14)
    ax.axvline(x=150, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Only set Y label for left plots, X label for bottom plots (due to sharex/sharey)
    if i % 2 == 0: ax.set_ylabel("Steps")
    if i >= 2: ax.set_xlabel("Epochs")

    # Set limits
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)

plt.tight_layout()
plt.savefig("/home/citic_gii/iwinac2026/plot_comparison_grid.png", dpi=300)
plt.show()