import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import re

def get_data_stats(filepath):
    epochs, steps, methods = [], [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                epoch = int(parts[0])
                method = str(parts[1])
                step  = int(parts[2])
            except (ValueError, IndexError):
                continue
            epochs.append(epoch)
            steps.append(step)
            methods.append(method)
    return epochs, steps, methods


def get_goal_data(filepath):
    epochs, goal_ids, competences = [], [], []
    goal_map = {}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                epoch = int(parts[0])
            except ValueError:
                continue

            vec_match = re.search(r'\[([^\]]+)\]', line)
            if vec_match is None:
                continue

            vec_str = " ".join(vec_match.group(1).split())
            comp_str = parts[-1]
            comp = float(comp_str) if comp_str not in ("None", "competence") else np.nan

            if vec_str not in goal_map:
                goal_map[vec_str] = len(goal_map)

            epochs.append(epoch)
            goal_ids.append(goal_map[vec_str])
            competences.append(comp)

    labels = {v: f"Goal {v+1}" for k, v in goal_map.items()}
    return np.array(epochs), np.array(goal_ids), np.array(competences, dtype=float), labels


# ── cesta k souboru ────────────────────────────────────────────────────────
filepath = "/Users/jakubmuller/Desktop/WORK/ijcr2026/fanuc_independent.txt"

e_cur, s_cur, m_cur = get_data_stats(filepath)
ep, goal_ids, competences, goal_labels = get_goal_data(filepath)

# ── layout ─────────────────────────────────────────────────────────────────
GOAL_COLORS = ["#C0392B", "#8E44AD", "#E67E22", "#16A085", "#2471A3"]
num_goals = len(goal_labels)
goal_colors_used = [GOAL_COLORS[i % len(GOAL_COLORS)] for i in range(num_goals)]

fig = plt.figure(figsize=(13, 10))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1.1], hspace=0.25)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════════════════
# SUBPLOT 1 – steps
# ═══════════════════════════════════════════════════════
method_colors = {
    'exploration': '#2980B9',        # Modrá
    'exploration_path': '#27AE60',   # Zelená
    'improvement': '#D35400'         # Tmavě oranžová
}
for i in range(len(e_cur) - 1):
    method = m_cur[i]
    color = method_colors.get(method, '#7F8C8D')  # výchozí šedá, pokud nenajde
    ax1.plot(e_cur[i:i+2], s_cur[i:i+2], color=color, linewidth=2)

ax1.set_ylabel("Steps", fontsize=14)
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 120)
ax1.set_xticks(np.arange(0, 91, 10))
ax1.set_yticks(np.arange(0, 121, 10))
ax1.tick_params(labelsize=14)
ax1.grid(linestyle="-", alpha=0.3)
legend_elements = [
    Line2D([0], [0], color='#2980B9', lw=2, label='Exploration'),
    Line2D([0], [0], color='#27AE60', lw=2, label='Exploration Path'),
    Line2D([0], [0], color='#D35400', lw=2, label='Improvement')
]
ax1.legend(handles=legend_elements, loc="center right", fontsize=14)
ax1.axvline(x=50, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.text(52, 110, 'Button position change', fontsize=16)

# ═══════════════════════════════════════════════════════
# SUBPLOT 2 – selected goal + competence
# ═══════════════════════════════════════════════════════
    
valid = ~np.isnan(competences)

# 1. Nejprve vykreslíme čáru competence
ax2.plot(ep[valid], competences[valid],
         color="#2C3E50", linewidth=1.8, linestyle="--", alpha=0.75, zorder=1)

# 2. Poté vykreslíme body pro zvolené cíle přímo na hodnoty competence
for gid in range(num_goals):
    mask = (goal_ids == gid) & valid # Ujistíme se, že máme pouze platné hodnoty competence
    if not mask.any():
        continue
    # y-ová souřadnice je teď "competences[mask]" místo fixní "goal_ids[mask]"
    ax2.scatter(ep[mask], competences[mask],
                color=goal_colors_used[gid], s=50, zorder=3, alpha=0.85)
    
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Competence", fontsize=14, color="#2C3E50")
ax2.tick_params(axis="y", labelsize=14, labelcolor="#2C3E50")
ax2.set_xlabel("Epochs", fontsize=14)
ax2.tick_params(axis="x", labelsize=14)
ax2.set_xticks(np.arange(0, 91, 10))
ax2.grid(axis="both", linestyle="-", alpha=0.25) # Přidána mřížka i pro osu y
ax2.axvline(x=50, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

# for gid in range(num_goals):
#     mask = goal_ids == gid
#     if not mask.any():
#         continue
#     ax2.scatter(ep[mask], goal_ids[mask],
#                 color=goal_colors_used[gid], s=50, zorder=3, alpha=0.85)

# ax2_comp = ax2.twinx()
# valid = ~np.isnan(competences)
# ax2_comp.plot(ep[valid], competences[valid],
#               color="#2C3E50", linewidth=1.8, linestyle="--", alpha=0.75)
# ax2_comp.set_ylim(0, 1.15)
# ax2_comp.set_xlim(0, 90)
# ax2_comp.set_ylabel("Competence", fontsize=14, color="#2C3E50")
# ax2_comp.tick_params(axis="y", labelsize=14, labelcolor="#2C3E50")

# ax2.set_yticks(list(range(num_goals)))
# ax2.set_yticklabels([goal_labels[i] for i in range(num_goals)], fontsize=10)
# ax2.set_xlim(0, 90)
# ax2.set_ylim(-0.6, num_goals - 0.4)
# ax2.set_xlabel("Epochs", fontsize=14)
# ax2.set_ylabel("Selected goal", fontsize=14)
# ax2.tick_params(axis="both", labelsize=14)
# ax2.set_xticks(np.arange(0, 91, 10))
# ax2.grid(axis="x", linestyle="-", alpha=0.25)
# ax2.axvline(x=50, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

goal_patches = [mpatches.Patch(color=goal_colors_used[i], label=goal_labels[i])
                for i in range(num_goals)]
comp_line = plt.Line2D([0], [0], color="#2C3E50", linewidth=1.8,
                       linestyle="--", label="Competence")
ax2.legend(handles=goal_patches + [comp_line], fontsize=14,
           loc="center right", framealpha=0.85)

plt.savefig("/Users/jakubmuller/Desktop/WORK/IJCR2026/plot_fanuc_goals.png", dpi=300, bbox_inches="tight")
plt.show()