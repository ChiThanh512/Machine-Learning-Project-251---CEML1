import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100).T
        return mfccs_features
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

def read_dataset_and_save_feture(root_folder_path, save_path):
    features_list = []
    labels_list = []
    
    if not os.path.exists(root_folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{root_folder_path}'")
        return

    for label in tqdm(os.listdir(root_folder_path), desc="Processing Labels"):
        folder_path = os.path.join(root_folder_path, label)
        if not os.path.isdir(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

    # Lưu dưới dạng object array để chứa các chuỗi có độ dài khác nhau
    np.savez_compressed(save_path, features=np.array(features_list, dtype=object), labels=np.array(labels_list))
    print(f"\nTrích xuất hoàn tất! Đã lưu {len(features_list)} chuỗi đặc trưng.")

def load_and_preprocess_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    X_list = data['features']
    y_str = data['labels']

    # Chuẩn hóa đặc trưng cho từng chuỗi
    scaler = StandardScaler()
    # Nối tất cả các chuỗi lại để fit scaler một lần duy nhất
    X_concatenated = np.vstack(X_list)
    scaler.fit(X_concatenated)
    # Áp dụng scaler cho từng chuỗi riêng lẻ
    X_scaled_list = [scaler.transform(x) for x in X_list]

    # Mã hóa nhãn (giữ nguyên như cũ)
    label_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    class_names = sorted(label_map, key=label_map.get)
    y_encoded = np.array([label_map[label] for label in y_str])

    return X_scaled_list, y_encoded, class_names

def split_train_test(X_list, y, test_size=0.2, random_state=42):
    # Cần chuyển X_list thành mảng tạm để stratify hoạt động
    indices = np.arange(len(X_list))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y)
    
    X_train = [X_list[i] for i in train_indices]
    X_test = [X_list[i] for i in test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test