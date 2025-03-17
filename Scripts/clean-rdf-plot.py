import numpy as np
import os
import matplotlib.pyplot as plt

bright_colors = ['xkcd:blue', 'xkcd:orange','xkcd:green','xkcd:light blue','xkcd:light orange','xkcd:light green','xkcd:brown','xkcd:light brown'] # This is a part of the ColorBrewer Set1 palette

def set_plot_style(ax=None):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 32
    plt.tight_layout()
    if ax is None:
        ax = plt.gca()  # get the current axes if not provided
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=24, length=10, width=3)

# Define paths and files
# Use the current directory the script is in
base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = base_path

# Plot for Bicarb data with focused x-range
plt.figure(figsize=(10, 8))

# Load and plot first Bicarb dataset
data = np.loadtxt(os.path.join(base_path, 'gofrBicarbCOnorm.dat'))
r = data[:, 0]
rdf = data[:, 1] 
# No smoothing applied
plt.plot(r, rdf, label='Bicarb-CO (norm)', linewidth=5, color=bright_colors[0])

# Load and plot second Bicarb dataset
data = np.loadtxt(os.path.join(base_path, 'gofrBicarbCOOH.dat'))
r = data[:, 0]
rdf = data[:, 1]
# No smoothing applied
plt.plot(r, rdf, label='Bicarb-COOH', linewidth=5, color=bright_colors[1])

# Set axis limits and labels with zoomed x-range (1 to 2)
plt.xlim([1, 2])
plt.ylim([0, 20])
plt.xticks(np.arange(1, 2.01, 0.2))  # Adjusted tick spacing for the smaller range
plt.yticks(np.arange(0, 20.01, 5))
plt.xlabel('$r$ [$\mathrm{\AA}$]', fontsize=32)
plt.ylabel('RDF', fontsize=32)
plt.legend(['C-O distance', 'C-OH distance'], loc='upper right', fontsize=24, frameon=False)
set_plot_style()
plt.savefig(os.path.join(output_dir, "gofrBicarb_comparison.pdf"), format='pdf', dpi=1000)
plt.show()
