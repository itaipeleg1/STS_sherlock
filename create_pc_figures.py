import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import glob

# The two lists
top_100 = np.array([983, 982, 989, 1019, 981, 986, 505, 504, 170, 800, 506, 503, 1012, 784,
                        507, 502, 980, 199, 200, 198, 1664, 1009, 1013, 1015, 122, 764, 501, 1004,
                        1022, 1020, 1016, 1021, 172, 1674, 491, 1579, 987, 790, 1663, 957, 234, 1662,
                        121, 127, 1408, 201, 1002, 1669, 99, 1407, 490, 1402, 173, 120, 1668, 1803,
                        1403, 990, 1670, 765, 1413, 1158, 251, 1804, 496, 1802, 1010, 1791, 1657, 1665,
                        372, 169, 1667, 1404, 100, 1401, 997, 968, 323, 1406, 985, 98, 1412, 197,
                        252, 781, 1405, 145, 123, 489, 1388, 956, 1411, 1646, 1671, 1005, 1400, 1397,
                        1393, 994])

bottom_100 = np.array([1938, 524, 519, 1945, 1842, 133, 1963, 548, 1946, 922, 518, 475, 1950, 306,
                            1936, 1948, 923, 130, 696, 1175, 948, 1843, 1964, 924, 708, 1949, 1943, 1935,
                            1677, 711, 111, 859, 709, 303, 1941, 1942, 722, 291, 105, 1940, 710, 1933,
                            713, 1965, 302, 1962, 85, 712, 1937, 1934, 947, 84, 292, 945, 1932, 293,
                            304, 295, 110, 108, 946, 517, 1931, 480, 514, 1961, 509, 294, 296, 109,
                            714, 1930, 516, 513, 1927, 479, 515, 1929, 1928, 1653, 1652, 298, 1655, 1926,
                            1676, 715, 1654, 106, 512, 510, 107, 716, 131, 511, 132, 717, 718, 719,
                            721, 720])

frames_dir = "/home/new_storage/sherlock/data/frames"
output_dir = "/home/new_storage/sherlock/STS_sherlock"

def get_random_image_from_folder(index):
    """Get a random image from the TR#### folder corresponding to the index."""
    folder_name = f"TR{index:04d}"
    folder_path = os.path.join(frames_dir, folder_name)

    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return None

    # Get all image files (common image extensions)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_files:
        print(f"Warning: No images found in {folder_path}")
        return None

    # Randomly select one image
    selected_image = random.choice(image_files)
    return Image.open(selected_image)

def create_figure(indices, title, output_filename, rows=5, cols=10):
    """Create a figure with images arranged in a grid."""
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(title, fontsize=16)

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = get_random_image_from_folder(idx)

        if img is not None:
            axes[i].imshow(img)
            axes[i].set_title(f"TR{idx:04d}", fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f"TR{idx:04d}\nNot Found",
                        ha='center', va='center', fontsize=10)

        axes[i].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

# Create figures for top 100
print("Creating figures for top 100...")
create_figure(top_100[:50], "Top 100 PC1 - Part 1 (Images 1-50)", "top_100_part1.png")
create_figure(top_100[50:], "Top 100 PC1 - Part 2 (Images 51-100)", "top_100_part2.png")

# Create figures for bottom 100
print("Creating figures for bottom 100...")
create_figure(bottom_100[:50], "Bottom 100 PC1 - Part 1 (Images 1-50)", "bottom_100_part1.png")
create_figure(bottom_100[50:], "Bottom 100 PC1 - Part 2 (Images 51-100)", "bottom_100_part2.png")

print("All figures created successfully!")
