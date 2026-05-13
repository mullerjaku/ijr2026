import matplotlib.pyplot as plt
import numpy as np
import os

def get_data_stats(base_path, num_files=20):
    steps_data = {}
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        epoch = int(parts[0])
                        step = int(parts[2])
                        if epoch not in steps_data:
                            steps_data[epoch] = []
                        steps_data[epoch].append(step)
        except FileNotFoundError:
            pass 
            
    epochs_avg = []
    steps_avg = []
    steps_p25 = []
    steps_p75 = []

    for epoch in sorted(steps_data.keys()):
        steps_for_this_epoch = steps_data[epoch]
        median_step = np.median(steps_for_this_epoch)
        p25 = np.percentile(steps_for_this_epoch, 25)
        p75 = np.percentile(steps_for_this_epoch, 75)

        epochs_avg.append(epoch)
        steps_avg.append(median_step)
        steps_p25.append(p25)
        steps_p75.append(p75)
        
    return epochs_avg, steps_avg, steps_p25, steps_p75

def remove_gap_and_shift(epochs, values, p25, p75):
    """
    Odstraní data mezi 100 a 150.
    Data >= 150 posune o 50 doleva (takže 150 -> 100, 151 -> 101...).
    """
    new_e, new_v, new_p25, new_p75 = [], [], [], []
    
    for e, v, p2, p7 in zip(epochs, values, p25, p75):
        if e < 100:
            new_e.append(e)
            new_v.append(v)
            new_p25.append(p2)
            new_p75.append(p7)
        elif e >= 150:
            new_e.append(e - 50) 
            new_v.append(v)
            new_p25.append(p2)
            new_p75.append(p7)
            
    return new_e, new_v, new_p25, new_p75

num_files_cur = 20
paths_1 = {
    "cur": "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_",
    "nov": "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/nov_"
}
paths_2 = {"cur": "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_wm_"}
paths_3 = {"cur": "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_comb_"}

fig, ax = plt.subplots(figsize=(12, 8))

e_nov, s_nov, p25_nov, p75_nov = get_data_stats(paths_1["nov"], num_files_cur)
e_cur1, s_cur1, p25_cur1, p75_cur1 = get_data_stats(paths_1["cur"], num_files_cur)
e_cur2, s_cur2, p25_cur2, p75_cur2 = get_data_stats(paths_2["cur"], num_files_cur)
e_cur3, s_cur3, p25_cur3, p75_cur3 = get_data_stats(paths_3["cur"], num_files_cur)

e_nov, s_nov, p25_nov, p75_nov = remove_gap_and_shift(e_nov, s_nov, p25_nov, p75_nov)
e_cur1, s_cur1, p25_cur1, p75_cur1 = remove_gap_and_shift(e_cur1, s_cur1, p25_cur1, p75_cur1)
e_cur2, s_cur2, p25_cur2, p75_cur2 = remove_gap_and_shift(e_cur2, s_cur2, p25_cur2, p75_cur2)
e_cur3, s_cur3, p25_cur3, p75_cur3 = remove_gap_and_shift(e_cur3, s_cur3, p25_cur3, p75_cur3)

x_points = [0, 100, 100, 450]
y_points = [4, 4, 200, 200]

ax.plot(x_points, y_points, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Classical Industrial Robot')

ax.axvline(x=100, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(70, 210, 'WM and / or Goal position change', fontsize=16)

ax.plot(e_nov, s_nov, color='#2ca02c', linestyle='-.', linewidth=2, label='Goal Change (Nov. Baseline)')
ax.fill_between(e_nov, p25_nov, p75_nov, color='#2ca02c', alpha=0.1)

ax.plot(e_cur1, s_cur1, color='#1f77b4', linestyle='-.', linewidth=2, label='Goal Change (Mot. Engine)')
ax.fill_between(e_cur1, p25_cur1, p75_cur1, color='#1f77b4', alpha=0.2)

ax.plot(e_cur2, s_cur2, color='#ff7f0e', linestyle='-', linewidth=2, label='WM Change (Mot. Engine)')
ax.fill_between(e_cur2, p25_cur2, p75_cur2, color='#ff7f0e', alpha=0.1)

ax.plot(e_cur3, s_cur3, color='#d62728', linestyle='-', linewidth=2, label='Combined Change (Mot. Engine)')
ax.fill_between(e_cur3, p25_cur3, p75_cur3, color='#d62728', alpha=0.1)

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Steps", fontsize=16)
ax.grid(linestyle='-', alpha=0.3)

ax.set_xlim(0, 200)
ax.set_ylim(0, 205)

x_ticks = np.arange(0, 201, 25)
y_ticks = np.arange(0, 201, 25)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.tick_params(axis='both', which='major', labelsize=16)

ax.legend(loc='upper left', fontsize=16)

plt.tight_layout()
plt.savefig("/Users/jakubmuller/Desktop/WORK/iwinac2026/plot_ijrc_combined.png", dpi=300)
plt.show()