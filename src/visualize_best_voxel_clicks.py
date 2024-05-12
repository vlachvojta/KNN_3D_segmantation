import os
import numpy as np
from matplotlib import pyplot as plt
import csv

# Load the results from the file
# DIR='../results/testing_voxel_click_mini_mini'
DIR='../results/testing_voxel_click_mini'
RESULT_FILE = os.path.join(DIR, 'result_summary.txt')
print(f'Loading results from {RESULT_FILE}')

# read RESULT_FILE as is and print
# with open(RESULT_FILE) as f:
#     print(f.read())

# read RESULT_FILE as csv and print. ignore headers!!
with open(RESULT_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    rows = [row for row in reader if row[2] != 'None'][1:]

for row in rows:
    print(row)

results = np.array(rows, dtype=float)
voxel_sizes = results[:,0]
click_areas = results[:,1]
ious = results[:,2]

# Plot the results as a heatmap (x = click_area, y = voxel_size, color = iou)
plt.figure()
plt.scatter(click_areas, voxel_sizes, c=ious, cmap='viridis')
plt.xlabel('Click Area')
plt.ylabel('Voxel Size')
plt.title('IoU')
plt.colorbar()
plt.tight_layout()
out_file = os.path.join(DIR, 'heatmap.png')
plt.savefig(out_file)
plt.clf()
print(f'Heatmap saved to {out_file}')