import os
import numpy as np
from matplotlib import pyplot as plt
import csv

# Load the results from the file
DIR='../results/testing_voxel_click_mini'
RESULT_FILE = os.path.join(DIR, 'result_summary.txt')
print(f'Loading results from {RESULT_FILE}')

# read RESULT_FILE as csv and print. ignore headers!!
with open(RESULT_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    rows = [row for row in reader if row[2] != 'None']
    if rows[0][0] == 'voxel_size':
        rows = rows[1:]

    rows = [[float(val) for val in row] for row in rows]
    rows = [[voxel_size, click_area, round(IOU, 2)] for voxel_size, click_area, IOU in rows if IOU != 0]

for row in rows:
    print(row)

data_orig = rows

def get_click_voxel_ratio(click_area, voxel_size):
    return round(click_area / voxel_size, 1)

# data with voxel_click ratio:
data = [(get_click_voxel_ratio(click_area, voxel_size), voxel_size, IOU) for voxel_size, click_area, IOU in data_orig]

# Arrange data by voxel_click_ratio and voxel_size
data_dict = {}
for voxel_click_ratio, voxel_size, IOU in data:
    if voxel_click_ratio not in [1, 1.5, 2]:
        continue
    if voxel_click_ratio not in data_dict:
        data_dict[voxel_click_ratio] = {}
    data_dict[voxel_click_ratio][voxel_size] = IOU

# Sort the data by voxel_size
data_dict_sorted = {}
for voxel_click_ratio, results in data_dict.items():
    # results = sorted(results, key=lambda x: x[0])
    results = [results[voxel_size] for voxel_size in sorted(results.keys())]
    # results = [round(IOU, 2) for voxel_size, IOU in results]
    data_dict_sorted[f'{voxel_click_ratio}*voxel_size'] = results

print('data dict sorted:')
for k, v in data_dict_sorted.items():
    print(k, v)

ratios = list(map(str, np.unique([voxel_size for voxel_size, _, _ in data_orig])))
ratio_restults = data_dict_sorted

"""
Data should look something like this:
ratios = ("0.01", "0.02", "0.03", "0.04")
ratio_restults = {
    '1x': (18.35, 18.43, 14.98, 17.75),
    '1.5x': (38.79, 48.83, 47.50, 22.22),
    '2x': (189.95, 195.82, 217.19, 34.88),
}
"""

# Prepare plot
x = np.arange(len(ratios))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in ratio_restults.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean IOU')
ax.set_title('Effect of voxel size and click area on mean IOU in S3DIS dataset.')
ax.set_xticks(x + width, ratios)
ax.set_xlabel('Voxel size')
ax.legend(loc='upper left', ncols=3)
plt.tight_layout()
ax.set_ylim(0, max([max(measurement) for measurement in ratio_restults.values()]) + 10)

# Save the plot
out_file_png = os.path.join(DIR, 'voxel_click_IOU.png')
out_file_svg = os.path.join(DIR, 'voxel_click_IOU.svg')
plt.savefig(out_file_png)
plt.savefig(out_file_svg)
plt.clf()
print(f'Groupbar chart saved to {out_file_png} and {out_file_svg}')
