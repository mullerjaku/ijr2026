import matplotlib.pyplot as plt
import numpy as np

num_files_cur = 20
base_path_cur = "/home/citic_gii/iwinac2026/results_data/cur_"
base_path_nov = "/home/citic_gii/iwinac2026/results_data/nov_"
steps_data_cur = {}
steps_data_nov = {}


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
    filename = f"{base_path_nov}{i}.txt"
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
                    if epoch not in steps_data_nov:
                        steps_data_nov[epoch] = []
                    
                    # Přidáme hodnotu 'step' ze souboru
                    steps_data_nov[epoch].append(step)
    except FileNotFoundError:
        print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
        
epochs_nov_avg = []
steps_nov_avg = []
steps_nov_p25 = []
steps_nov_p75 = []


for epoch in sorted(steps_data_nov.keys()):
    steps_for_this_epoch = steps_data_nov[epoch]

    if len(steps_for_this_epoch) < num_files_cur:
        print(f"Poznámka: Epocha {epoch} má jen {len(steps_for_this_epoch)}/{num_files_cur} záznamů.")

    median_step = np.median(steps_for_this_epoch)
    p25 = np.percentile(steps_for_this_epoch, 25)
    p75 = np.percentile(steps_for_this_epoch, 75)

    epochs_nov_avg.append(epoch)
    steps_nov_avg.append(median_step)
    steps_nov_p25.append(p25)
    steps_nov_p75.append(p75)

print("Průměrování 'vystup_nov' dokončeno.")



plt.figure(figsize=(10, 6))


x_points = [0, 150, 150, 500]
y_points = [4, 4, 200, 200]
plt.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
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

plt.plot(epochs_nov_avg, steps_nov_avg, color='green', linestyle='-', alpha=0.6, label='Novelty')
plt.fill_between(
    epochs_nov_avg,
    steps_nov_p25,
    steps_nov_p75,
    color='green',
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
plt.savefig("/home/citic_gii/iwinac2026/plot_ijrc.png", dpi=300)
plt.show()