import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict

# ── Parse ────────────────────────────────────────────────────────────────────

def parse_line(line):
    """
    Returns (pred, real, step, epoch) or (None, None, step, epoch) for fail lines.
    pred / real are numpy arrays of shape (3,).
    """
    line = line.strip()
    if not line:
        return None

    # Extract the two trailing integers (step, epoch)
    tail = re.search(r'(\d+)\s+(\d+)\s*$', line)
    if tail is None:
        return None
    step  = int(tail.group(1))
    epoch = int(tail.group(2))

    if line.startswith('None'):
        return (None, None, step, epoch)

    # Extract all floats that appear inside brackets
    arrays = re.findall(r'\[([^\]]+)\]', line)
    if len(arrays) < 2:
        return None

    pred = np.array([float(x) for x in arrays[0].split()])
    real = np.array([float(x) for x in arrays[1].split()])
    return (pred, real, step, epoch)


def load_data(path):
    deviations = defaultdict(list)   # epoch -> [euclidean distances]
    fails      = defaultdict(int)    # epoch -> count of None lines

    with open(path, 'r') as fh:
        for raw in fh:
            parsed = parse_line(raw)
            if parsed is None:
                continue
            pred, real, step, epoch = parsed
            if pred is None:
                fails[epoch] += 1
            else:
                dist = np.linalg.norm(pred - real)
                deviations[epoch].append(dist)

    return deviations, fails


# ── Aggregate ────────────────────────────────────────────────────────────────

def aggregate(deviations, fails, all_epochs):
    medians, p25, p75, fail_counts = [], [], [], []
    for ep in all_epochs:
        devs = deviations.get(ep, [])
        if devs:
            medians.append(np.median(devs))
            p25.append(np.percentile(devs, 25))
            p75.append(np.percentile(devs, 75))
        else:
            medians.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
        fail_counts.append(fails.get(ep, 0))
    return (np.array(medians), np.array(p25), np.array(p75),
            np.array(fail_counts))


# ── Plot ─────────────────────────────────────────────────────────────────────

# def plot(path='pos_independent.txt', out='deviation_plot.png'):
#     deviations, fails = load_data(path)

#     all_epochs = sorted(set(deviations) | set(fails))
#     epochs = np.array(all_epochs)

#     medians, p25, p75, fail_counts = aggregate(deviations, fails, all_epochs)

#     # ── figure layout ─────────────────────────────────────────────────────────
#     plt.figure(figsize=(9, 4))
#     plt.grid(linestyle='-', alpha=0.3)
#     plt.tick_params(axis='both', labelsize=14)

#     plt.xlim(0,200)
#     plt.ylim(0,0.08)
#     plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))

#     mask1 = epochs <= 100
#     mask2 = epochs > 100

#     # Získání počtu fails v epoše 100
#     fail_100 = fails.get(100, 0)
#     fail_101 = fails.get(101, 0)

#     # ── První část: do epochy 100 (modrá barva) ───────────────────────────────
#     color1 = '#5b9bd5'
#     plt.fill_between(epochs[mask1], p25[mask1], p75[mask1],
#                      alpha=0.3, color=color1)
#     plt.plot(epochs[mask1], medians[mask1],
#              color=color1, linewidth=2.2, label='Predefined WM')

#     # ── Druhá část: od epochy 100 (oranžová barva) ────────────────────────────
#     color2 = '#ed7d31' # Typická doplňková oranžová
#     plt.fill_between(epochs[mask2], p25[mask2], p75[mask2],
#                      alpha=0.3, color=color2)
#     plt.plot(epochs[mask2], medians[mask2],
#              color=color2, linewidth=2.2, label='Learned WM')
#     plt.axvline(x=100, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

#     plt.text(100, 0.082, f'WM change', 
#              ha='center', va='bottom', fontsize=14, 
#              color='black', clip_on=False)
    
#     bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
    
#     plt.text(100, 0.082, f'WM change\nFails: {fail_100} $\\rightarrow$ {fail_101}', 
#              ha='center', va='bottom', fontsize=12, 
#              bbox=bbox_props, clip_on=False)

#     plt.ylabel('Euclidean deviation', fontsize=14)
#     plt.xlabel('Epochs', fontsize=14)
#     plt.legend(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     print(f'Saved → {out}')
#     plt.show()

# ── Plot ─────────────────────────────────────────────────────────────────────

def plot(path='pos_independent.txt', out='deviation_plot.png'):
    deviations, fails = load_data(path)

    all_epochs = sorted(set(deviations) | set(fails))
    epochs = np.array(all_epochs)

    medians, p25, p75, fail_counts = aggregate(deviations, fails, all_epochs)

    # ── figure layout (přechod na objektový přístup s ax1) ────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.grid(linestyle='-', alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 0.08)
    ax1.yaxis.set_major_locator(MultipleLocator(0.01))

    mask1 = epochs <= 100
    mask2 = epochs > 100

    # ── První osa Y (Levá): Odchylky ──────────────────────────────────────────
    color1 = '#5b9bd5'
    ax1.fill_between(epochs[mask1], p25[mask1], p75[mask1], alpha=0.3, color=color1)
    ax1.plot(epochs[mask1], medians[mask1], color=color1, linewidth=2.2, label='Initial WM')

    color2 = '#ed7d31'
    ax1.fill_between(epochs[mask2], p25[mask2], p75[mask2], alpha=0.3, color=color2)
    ax1.plot(epochs[mask2], medians[mask2], color=color2, linewidth=2.2, label='Adapted WM')

    ax1.set_ylabel('Euclidean deviation', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=14)

    # ── Druhá osa Y (Pravá): Faily ────────────────────────────────────────────
    # Vytvoření sdílené osy X
    ax2 = ax1.twinx() 

    fail_100 = fails.get(100, 0)
    fail_101 = fails.get(101, 0)
    
    # Vykreslíme faily jako poloprůhledné červené sloupečky na pozadí
    bar_color = '#9b59b6'
    bars = ax2.bar([100, 101], [fail_100, fail_101], color=bar_color, alpha=0.6, edgecolor='black', linewidth=1.5, width=1.0, label='Fails count')
    ax2.bar_label(bars, padding=3, color='black', fontsize=12, fontweight='bold')
    
    # Nastavení pravé osy
    ax2.set_ylabel('Fails count', fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    
    # Omezíme pravou osu, aby sloupečky nešly až úplně nahoru a nepřekážely textu
    max_fails = max(fail_counts) 
    step = int(np.ceil((max_fails * 1.2) / 8 / 10) * 10)
    if step == 0: step = 5 # Pojistka pro extrémně malá čísla
    
    right_max = step * 8
    ax2.set_ylim(0, right_max)
    
    # Vynutíme přesně 9 ticků od 0 do right_max, takže se dokonale sejdou s levou mřížkou
    ax2.set_yticks(np.linspace(0, right_max, 9))

    # ── Přidání vizuálního oddělovače a textu ────────────────────────────────
    ax1.axvline(x=100, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    # Text nahoru nad čáru (tentokrát bez textu failů, protože už jsou v grafu)
    ax1.text(100, 0.082, 'WM change', ha='center', va='bottom', 
             fontsize=14, clip_on=False)

    # ── Sjednocení legendy ze dvou os ─────────────────────────────────────────
    # Získáme popisky z obou os a spojíme je do jedné legendy
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=14)

    # ── Dokončení ─────────────────────────────────────────────────────────────
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else 'pos_independent_2.txt'
    out = sys.argv[2] if len(sys.argv) > 2 else 'deviation_plot.png'
    plot(txt, out)