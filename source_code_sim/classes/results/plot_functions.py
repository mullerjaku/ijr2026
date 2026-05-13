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



plt.figure(figsize=(10, 6))


x_points = [0, 150, 150, 500]
y_points = [4, 4, 200, 200]
plt.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
plt.text(150 + 2, 205, 'Goal position change', fontsize=12)


plt.plot(epochs_cur_avg, steps_cur_avg, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
plt.fill_between(
    epochs_cur_avg,
    steps_cur_p25,
    steps_cur_p75,
    color='blue',
    alpha=0.2
)

plt.plot(epochs_0_avg, steps_0_avg, color='green', linestyle='-', alpha=0.6, label='Lin')
plt.fill_between(
    epochs_0_avg,
    steps_0_p25,
    steps_0_p75,
    color='green',
    alpha=0.2
)

plt.plot(epochs_1_avg, steps_1_avg, color='orange', linestyle='-', alpha=0.6, label='SVR')
plt.fill_between(
    epochs_1_avg,
    steps_1_p25,
    steps_1_p75,
    color='orange',
    alpha=0.2
)

plt.plot(epochs_2_avg, steps_2_avg, color='yellow', linestyle='-', alpha=0.6, label='Grad')
plt.fill_between(
    epochs_2_avg,
    steps_2_p25,
    steps_2_p75,
    color='yellow',
    alpha=0.2
)

plt.plot(epochs_3_avg, steps_3_avg, color='grey', linestyle='-', alpha=0.6, label='MLP')
plt.fill_between(
    epochs_3_avg,
    steps_3_p25,
    steps_3_p75,
    color='grey',
    alpha=0.2
)

plt.xlabel("Epoch")
plt.ylabel("Steps")
plt.grid(linestyle='-', alpha=0.2) 
x_ticks = np.arange(0, 501, 25)
y_ticks = np.arange(0, 201, 25)
plt.xticks(x_ticks, fontsize=12)
plt.yticks(y_ticks, fontsize=12)
plt.xlim(0, 250)
plt.ylim(0, 250) 

plt.legend()
plt.tight_layout()
plt.savefig("/home/citic_gii/iwinac2026/plot_ijrc_um_functions.png", dpi=300)
plt.show()