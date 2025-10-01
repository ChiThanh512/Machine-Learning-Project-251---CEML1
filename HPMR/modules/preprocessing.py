import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def extract_features(file):
    # Load audio and sample rate of audio
    audio,sample_rate = librosa.load(file)
    # Extract features using mel-frequency coefficient
    extracted_features = librosa.feature.mfcc(y=audio,
                                              sr=sample_rate,
                                              n_mfcc=40)
    
    # Scale the extracted features
    extracted_features = np.mean(extracted_features.T,axis=0)
    # Return the extracted features
    return extracted_features
    

def read_dataset_and_save_feture(root_folder_path, save_path):
    """
    Tạo dataset và lưu kết quả dưới dạng file .npz nén.
    - root_folder_path: Đường dẫn đến thư mục 'spoken_digit_data'.
    - save_path: Đường dẫn để lưu file .npz.
    """
    dataset = []
    
    if not os.path.exists(root_folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{root_folder_path}'")
        return

    for label in tqdm(os.listdir(root_folder_path), desc="Processing Labels"):
        folder_path = os.path.join(root_folder_path, label)
        if not os.path.isdir(folder_path):
            continue
            
        for file_name in tqdm(os.listdir(folder_path), desc=f"Files in {label}", leave=False):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                dataset.append([features, label])
    
    # Chuyển danh sách thành DataFrame để dễ xử lý
    df = pd.DataFrame(dataset, columns=['features', 'class'])
    
    # --- CHUYỂN ĐỔI VÀ LƯU DƯỚI DẠNG NUMPY ---
    # Chuyển cột 'features' (list các array) thành một mảng NumPy 2D
    X = np.array(df['features'].tolist())
    # Chuyển cột 'class' thành một mảng NumPy 1D
    y = np.array(df['class'].tolist())
    
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Lưu cả hai mảng X và y vào một file .npz nén
    np.savez_compressed(save_path, features=X, labels=y)
    
    print(f"\nTrích xuất hoàn tất! Đã tạo và lưu dataset tại: {save_path}")
    print(f"Kích thước mảng đặc trưng (X): {X.shape}")
    print(f"Kích thước mảng nhãn (y): {y.shape}")

def load_and_preprocess_data(data_path):
    """
    Tải dữ liệu từ file .npz, chuẩn hóa đặc trưng và mã hóa nhãn.

    Args:
        data_path (str): Đường dẫn đến file .npz.

    Returns:
        tuple: Một tuple chứa (X_scaled, y_encoded, class_names)
               - X_scaled: Mảng đặc trưng đã được chuẩn hóa.
               - y_encoded: Mảng nhãn đã được mã hóa thành số.
               - class_names: Danh sách tên các lớp gốc.
    """
    print(f"\n--- Đang tải và tiền xử lý dữ liệu từ '{data_path}' ---")
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{data_path}'")
        return None, None, None

    data = np.load(data_path, allow_pickle=True)
    X = data['features']
    y_str = data['labels']

    # 1. Chuẩn hóa đặc trưng (Feature Scaling)
    # Giúp các mô hình hội tụ nhanh hơn và hoạt động tốt hơn.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Đã chuẩn hóa đặc trưng (X).")

    # 2. Mã hóa nhãn (Label Encoding)
    # Chuyển nhãn dạng chữ ('one', 'two') thành dạng số (0, 1, ...).
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    class_names = label_encoder.classes_
    print(f"Đã mã hóa nhãn (y). Các lớp: {class_names}")

    return X_scaled, y_encoded, class_names

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành các tập huấn luyện và kiểm thử.

    Args:
        X (np.array): Mảng đặc trưng.
        y (np.array): Mảng nhãn.
        test_size (float): Tỷ lệ của tập kiểm thử.
        random_state (int): Hạt giống ngẫu nhiên để đảm bảo kết quả có thể tái tạo.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n--- Đang chia dữ liệu thành tập huấn luyện và kiểm thử... ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm thử: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test