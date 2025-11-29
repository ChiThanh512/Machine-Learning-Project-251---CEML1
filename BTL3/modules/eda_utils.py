import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def plot_label_distribution(image_labels, class_names, num_classes):
    """Vẽ biểu đồ phân phối số lượng ảnh theo từng lớp."""
    label_counts = pd.Series(image_labels).value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_classes), label_counts.values)
    plt.xticks(range(num_classes), class_names, rotation=90)
    plt.title("Phân phối số lượng ảnh theo lớp")
    plt.tight_layout()
    plt.show()


def analyze_image_sizes(image_paths, sample_size=300):
    """Thống kê kích thước (W, H) trung bình, nhỏ nhất và lớn nhất của ảnh."""
    sizes = []
    for p in image_paths[:min(sample_size, len(image_paths))]:
        with Image.open(p) as im:
            sizes.append(im.size)
    sizes = np.array(sizes)
    mean_size = sizes.mean(axis=0)
    min_size = sizes.min(axis=0)
    max_size = sizes.max(axis=0)

    print(f"Kích thước ảnh (W,H): mean {mean_size}, | min {min_size}, | max {max_size}")
    return mean_size, min_size, max_size


def analyze_image_channels(image_paths, sample_size=300):
    """Đếm số kênh và mode của ảnh (RGB, RGBA, L, CMYK, v.v.)"""
    mode_to_channels = {
        "1": 1, "L": 1, "P": 1, "LA": 2,
        "RGB": 3, "HSV": 3, "YCbCr": 3,
        "RGBA": 4, "CMYK": 4
    }

    channel_counts, mode_counts = {}, {}
    for p in image_paths[:min(sample_size, len(image_paths))]:
        try:
            with Image.open(p) as im:
                mode = im.mode
                ch = mode_to_channels.get(mode, len(im.getbands()))
                channel_counts[ch] = channel_counts.get(ch, 0) + 1
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        except Exception:
            continue

    total_checked = sum(channel_counts.values())
    print(f"Tổng số ảnh kiểm tra: {total_checked}")
    print("Đếm theo mode:", mode_counts)
    print("Đếm theo số kênh:", dict(sorted(channel_counts.items())))

    # Vẽ biểu đồ phân phối kênh màu
    if total_checked > 0:
        ks = sorted(channel_counts.keys())
        vs = [channel_counts[k] for k in ks]
        plt.figure(figsize=(6, 4))
        plt.bar([str(k) for k in ks], vs)
        plt.title("Phân phối số kênh màu (1 / 2 / 3 / 4 / ...)")
        plt.xlabel("Số kênh")
        plt.ylabel("Số ảnh")
        plt.tight_layout()
        plt.show()

    return mode_counts, channel_counts


def show_random_images(image_paths, image_labels, class_names, n=12, ncols=4):
    """Hiển thị ngẫu nhiên n ảnh từ dataset với nhãn tương ứng."""
    plt.figure(figsize=(10, 10))
    idxs = np.random.choice(len(image_paths), size=min(n, len(image_paths)), replace=False)
    nrows = int(np.ceil(n / ncols))

    for i, ix in enumerate(idxs, 1):
        plt.subplot(nrows, ncols, i)
        plt.axis("off")
        plt.title(class_names[image_labels[ix]])
        plt.imshow(Image.open(image_paths[ix]))

    plt.tight_layout()
    plt.show()


def run_full_eda(image_paths, image_labels, class_names):
    """Chạy toàn bộ pipeline EDA."""
    num_classes = len(class_names)

    print("=== PHÂN TÍCH DỮ LIỆU BAN ĐẦU (EDA) ===")
    plot_label_distribution(image_labels, class_names, num_classes)
    analyze_image_sizes(image_paths)
    analyze_image_channels(image_paths)
    show_random_images(image_paths, image_labels, class_names)













