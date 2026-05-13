import matplotlib.pyplot as plt
import numpy as np
import os

def get_data_stats(base_path, num_files=20):
    """
    Loads data from files, calculates median, p25, and p75.
    Returns lists: epochs_avg, steps_avg, steps_p25, steps_p75
    """
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
            print(f"Varování: Soubor {filename} nebyl nalezen a bude přeskočen.")
            
    epochs_avg = []
    steps_avg = []
    steps_p25 = []
    steps_p75 = []

    for epoch in sorted(steps_data.keys()):
        steps_for_this_epoch = steps_data[epoch]
        
        # Calculate stats
        median_step = np.median(steps_for_this_epoch)
        p25 = np.percentile(steps_for_this_epoch, 25)
        p75 = np.percentile(steps_for_this_epoch, 75)

        epochs_avg.append(epoch)
        steps_avg.append(median_step)
        steps_p25.append(p25)
        steps_p75.append(p75)
        
    return epochs_avg, steps_avg, steps_p25, steps_p75

# --- Configuration ---
num_files_cur = 20
# Define paths for the 3 different scenarios
paths_1 = {
    "cur": "/home/citic_gii/iwinac2026/results_data/cur_",
    "nov": "/home/citic_gii/iwinac2026/results_data/nov_"
}
paths_2 = {
    "cur": "/home/citic_gii/iwinac2026/results_data/cur_wm_"
    # "nov" was commented out in original script 2
}
paths_3 = {
    "cur": "/home/citic_gii/iwinac2026/results_data/cur_comb_"
    # "nov" was commented out in original script 3
}

# --- Plotting Setup ---
# Create 1 row, 3 columns. Adjust figsize as needed (width, height)
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

# Common data for the "Predefined Setup" (Red Line)
x_points = [0, 150, 150, 500]
y_points = [4, 4, 200, 200]

# ==========================
# PLOT 1: Standard (Cur + Nov)
# ==========================
ax = axes[0]

# Load Data
e_cur, s_cur, p25_cur, p75_cur = get_data_stats(paths_1["cur"], num_files_cur)
e_nov, s_nov, p25_nov, p75_nov = get_data_stats(paths_1["nov"], num_files_cur)

# Plot Predefined
ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 5, 205, 'Goal position change', fontsize=12)

# Plot Motivational Engine (Blue)
ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)

# Plot Novelty (Green) - Only in Plot 1
ax.plot(e_nov, s_nov, color='green', linestyle='-', alpha=0.6, label='Novelty')
ax.fill_between(e_nov, p25_nov, p75_nov, color='green', alpha=0.2)

# ==========================
# PLOT 2: WM (Cur only)
# ==========================
ax = axes[1]

# Load Data
e_cur, s_cur, p25_cur, p75_cur = get_data_stats(paths_2["cur"], num_files_cur)

# Plot Predefined
ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 5, 205, 'World Model change', fontsize=12)

# Plot Motivational Engine (Blue)
ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)

# ==========================
# PLOT 3: Combined (Cur only)
# ==========================
ax = axes[2]

# Load Data
e_cur, s_cur, p25_cur, p75_cur = get_data_stats(paths_3["cur"], num_files_cur)

# Plot Predefined
ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 5, 205, 'WM & Goal position change', fontsize=12)

# Plot Motivational Engine (Blue)
ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)


# ==========================
# Common Formatting Loop
# ==========================
x_ticks = np.arange(0, 501, 25)
y_ticks = np.arange(0, 201, 25)

for i, ax in enumerate(axes):
    ax.set_xlabel("Epochs", fontsize=12)
    if i == 0:
        ax.set_ylabel("Steps", fontsize=12) # Only label Y on the first plot
    
    ax.grid(linestyle='-', alpha=0.2)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig("/home/citic_gii/iwinac2026/plot_ijrc_next.png", dpi=300)
plt.show()


def get_data_stats(base_path, num_files=20):
    """
    Loads data from files, calculates median, p25, and p75.
    Returns lists: epochs_avg, steps_avg, steps_p25, steps_p75
    """
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
            # print(f"Varování: Soubor {filename} nebyl nalezen.") 
            pass
            
    epochs_avg = []
    steps_avg = []
    steps_p25 = []
    steps_p75 = []

    for epoch in sorted(steps_data.keys()):
        steps_for_this_epoch = steps_data[epoch]
        
        if not steps_for_this_epoch:
            continue

        median_step = np.median(steps_for_this_epoch)
        p25 = np.percentile(steps_for_this_epoch, 25)
        p75 = np.percentile(steps_for_this_epoch, 75)

        epochs_avg.append(epoch)
        steps_avg.append(median_step)
        steps_p25.append(p25)
        steps_p75.append(p75)
        
    return epochs_avg, steps_avg, steps_p25, steps_p75

# --- Configuration ---
num_files_cur = 20

# Define paths
path_cur_1 = "/home/citic_gii/iwinac2026/results_data/cur_"
path_nov_1 = "/home/citic_gii/iwinac2026/results_data/nov_"

path_cur_2 = "/home/citic_gii/iwinac2026/results_data/cur_wm_"
path_nov_2 = "/home/citic_gii/iwinac2026/results_data/nov_wm_" # path exists in logic, though maybe unused

path_cur_3 = "/home/citic_gii/iwinac2026/results_data/cur_comb_"
# path_nov_3 ignored

# --- Plotting Setup ---
# sharex=True aligns the x-axis and hides tick labels for the top plots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharex=True)

# Common data
x_points = [0, 150, 150, 500]
y_points = [4, 4, 200, 200]

# ==========================
# PLOT 1: Goal Position Change
# ==========================
ax = axes[0]

e_cur, s_cur, p25_cur, p75_cur = get_data_stats(path_cur_1, num_files_cur)
e_nov, s_nov, p25_nov, p75_nov = get_data_stats(path_nov_1, num_files_cur)

ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 2, 205, 'Goal position change', fontsize=12)

ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)

ax.plot(e_nov, s_nov, color='green', linestyle='-', alpha=0.6, label='Novelty')
ax.fill_between(e_nov, p25_nov, p75_nov, color='green', alpha=0.2)

ax.legend(loc='upper left')

# ==========================
# PLOT 2: World Model Change
# ==========================
ax = axes[1]

e_cur, s_cur, p25_cur, p75_cur = get_data_stats(path_cur_2, num_files_cur)

ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 2, 205, 'World Model change', fontsize=12)

ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)

ax.legend(loc='upper left')

# ==========================
# PLOT 3: Combined
# ==========================
ax = axes[2]

e_cur, s_cur, p25_cur, p75_cur = get_data_stats(path_cur_3, num_files_cur)

ax.plot(x_points, y_points, color='red', linestyle='-', alpha=0.4, label='Predefined Setup')
ax.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(150 + 2, 205, 'WM & Goal position change', fontsize=12)

ax.plot(e_cur, s_cur, color='blue', linestyle='-', alpha=0.6, label='Motivational Engine')
ax.fill_between(e_cur, p25_cur, p75_cur, color='blue', alpha=0.2)

ax.legend(loc='upper left')

# ==========================
# Common Formatting
# ==========================
x_ticks = np.arange(0, 501, 25)
y_ticks = np.arange(0, 201, 25)

for ax in axes:
    ax.set_ylabel("Steps", fontsize=12)
    ax.grid(linestyle='-', alpha=0.2) 
    
    # We set ticks for all, but sharex=True hides the labels for top ones automatically
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250) 

# Only set the X Label on the very last plot
axes[-1].set_xlabel("Epochs", fontsize=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1) # Reduces space between the plots slightly
plt.savefig("/home/citic_gii/iwinac2026/plot_ijrc_down.png", dpi=300)
plt.show()