import matplotlib.pyplot as plt
import pandas as pd
import io
import sys

# Read in CSV data for the timings
try:
    # Read the data from the CSV file into a pandas DataFrame
    df = pd.read_csv('timings.csv')
except FileNotFoundError:
    print("Error: 'timings.csv' not found.", file=sys.stderr)
    print("Please create the file in the same directory as the script and run again.", file=sys.stderr)
    sys.exit(1) # Exit the script if the file is not found


# Extracting column names for easier access
matrix_size_col = 'Matrix Size'
time_no_tc_col = 'Time for SGEMM without Tensor Cores'
time_with_tc_col = 'Time for SGEMM with Tensor Cores and mixed precision (Volta)'


# --- Plotting the data ---

# Create a figure and an axes object for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the time without Tensor Cores
ax.plot(df[matrix_size_col], df[time_no_tc_col], marker='o', linestyle='-', label='Without Tensor Cores')

# Plotting the time with Tensor Cores
ax.plot(df[matrix_size_col], df[time_with_tc_col], marker='s', linestyle='--', label='With Tensor Cores (Mixed Precision)')

# --- Styling the plot ---

# Adding a title to the plot
ax.set_title('SGEMM Performance: Tensor Cores vs. No Tensor Cores', fontsize=16)

# Adding labels to the x and y axes
ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)

# Using a logarithmic scale for the y-axis to better visualize the performance difference
ax.set_yscale('log')

# Adding a legend to identify the lines
ax.legend()

# Adding grid lines for better readability
ax.grid(True, which="both", ls="--", c='0.7')

# Display the plot
plt.tight_layout()
plt.show()