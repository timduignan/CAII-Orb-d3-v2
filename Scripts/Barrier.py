import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define color palette
bright_colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:light blue', 
                'xkcd:light orange', 'xkcd:light green', 'xkcd:brown', 'xkcd:light brown']

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

# Read the data
data = pd.read_csv('Barrier.csv')

# Print column names to debug
print("Available columns in the CSV file:", data.columns.tolist())

# Use the first column as x and second column as y
# This is safer than assuming specific column names
x_column = data.columns[0]  # First column 
y_column = data.columns[1]  # Second column

# Normalize GFN2-xTB data so that the first value is 0
first_value = data[y_column].iloc[0]
data[y_column] = data[y_column] - first_value

# Create the plot
plt.figure(figsize=(10, 8))
plt.plot(data[x_column], data[y_column], 'o-', linewidth=5, markersize=8, color=bright_colors[0])

# Set labels and title
plt.xlabel('Distance (Å)', fontsize=32)
plt.ylabel('Energy (kcal/mol)', fontsize=32)

# Apply the styling
set_plot_style()

# Save the figure
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'energy_barrier_plot.pdf'), format='pdf', dpi=1000)

plt.show()
