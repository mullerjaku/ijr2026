import matplotlib.pyplot as plt
import numpy as np
import os

def get_data_stats(base_path, num_files=5):
    steps_data = {}
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
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

def get_data_stats_hard(base_path, num_files=5):
    steps_data = {}
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    epoch = int(parts[0])
                    step = int(parts[1])
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


num_files_cur = 5
paths = {
    "fanuc": "/Users/jakubmuller/Desktop/WORK/ijcr2026/fanuc_",
    "hard": "/Users/jakubmuller/Desktop/WORK/ijcr2026/fanuc_hard_"
}

# 1. Načtení dat
e_cur1, s_cur1, p25_cur1, p75_cur1 = get_data_stats(paths["fanuc"], num_files_cur)
e_hard1, s_hard1, p25_hard1, p75_hard1 = get_data_stats_hard(paths["hard"], num_files_cur)

plt.figure(figsize=(12, 8))
plt.plot(e_cur1, s_cur1, color='blue', linewidth=2, label='Motivational Engine')
plt.fill_between(e_cur1, p25_cur1, p75_cur1, color='blue', alpha=0.2)

plt.plot(e_hard1, s_hard1, color='green', linewidth=2, label='Classical Industrial Robot')
plt.fill_between(e_hard1, p25_hard1, p75_hard1, color='green', alpha=0.2)
plt.axvline(x=50, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
plt.text(52, 110, 'Button position change', fontsize=16)

# 4. Formátování grafu
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Steps", fontsize=16)
plt.grid(linestyle='-', alpha=0.3)

# Limity os
plt.xlim(0, 90)
plt.ylim(0, 120) 

# Ticky na osách
x_ticks = np.arange(0, 91, 10)
y_ticks = np.arange(0, 121, 10)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.tick_params(axis='both', which='major', labelsize=16)

# Legenda
plt.legend(loc='upper left', fontsize=16)

# Uložení a zobrazení
plt.tight_layout()
plt.savefig("/Users/jakubmuller/Desktop/WORK/IJCR2026/plot_fanuc.png", dpi=300)
plt.show()