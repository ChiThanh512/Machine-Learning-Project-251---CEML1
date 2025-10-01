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
    
    
def create_dataset_from_folders(root_folder_path, save_path):
    """
    Tạo dataset từ cấu trúc thư mục mới và lưu kết quả.
    - root_folder_path: Đường dẫn đến thư mục 'spoken_digit_data'.
    - save_path: Đường dẫn để lưu file DataFrame kết quả.
    """
    dataset = []
    
    # Kiểm tra xem thư mục gốc có tồn tại không
    if not os.path.exists(root_folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{root_folder_path}'")
        return

    # Lặp qua các thư mục con (eight, five, four,...) - đây chính là các nhãn
    for label in tqdm(os.listdir(root_folder_path), desc="Processing Labels"):
        folder_path = os.path.join(root_folder_path, label)
        
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(folder_path):
            continue
            
        # Lặp qua từng file âm thanh trong thư mục nhãn
        for file_name in tqdm(os.listdir(folder_path), desc=f"Files in {label}", leave=False):
            file_path = os.path.join(folder_path, file_name)
            
            # Trích xuất đặc trưng từ file
            features = extract_features(file_path)
            
            # Chỉ thêm vào dataset nếu trích xuất thành công
            if features is not None:
                dataset.append([features, label])
    
    # Chuyển danh sách thành DataFrame
    df = pd.DataFrame(dataset, columns=['features', 'class'])
    
    # Lưu DataFrame vào file pickle (cách tốt nhất để lưu đối tượng numpy)
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)
    
    print(f"\nTrích xuất hoàn tất! Đã tạo và lưu dataset tại: {save_path}")
    print(f"Tổng số mẫu: {len(df)}")
    display(df.head())