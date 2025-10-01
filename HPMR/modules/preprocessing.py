def extract_features(file_path):
    """Trích xuất 40 đặc trưng MFCC từ một file âm thanh."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None
    

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