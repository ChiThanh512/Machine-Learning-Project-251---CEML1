

def create_dataframe_from_folders(data_path):
    """
    Quét thư mục dữ liệu có cấu trúc {digit_name}/{file}.wav và tạo DataFrame.

    Args:
        data_path (str): Đường dẫn đến thư mục gốc (ví dụ: 'spoken_digit_data').

    Returns:
        pd.DataFrame: DataFrame với các cột 'digit', 'digit_label', và 'path'.
    """
    metadata = []
    
    # Ánh xạ từ tên thư mục (chữ) sang số
    digit_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }

    # Lặp qua các thư mục con trong đường dẫn dữ liệu
    for digit_label in tqdm(os.listdir(data_path), desc="Đang quét thư mục"):
        if digit_label not in digit_map:
            continue  # Bỏ qua nếu tên thư mục không phải là một chữ số

        digit = digit_map[digit_label]
        digit_folder_path = os.path.join(data_path, digit_label)

        if not os.path.isdir(digit_folder_path):
            continue

        # Lặp qua các file .wav trong mỗi thư mục chữ số
        for file_name in os.listdir(digit_folder_path):
            if file_name.endswith('.wav'):
                full_path = os.path.join(digit_folder_path, file_name)
                metadata.append({
                    'digit': digit,
                    'digit_label': digit_label,
                    'path': full_path
                })

    df = pd.DataFrame(metadata)
    return df


def get_random_audio(df, digit=0):
    """
    Chọn và phát một file audio ngẫu nhiên từ DataFrame.
    Nếu 'digit' được cung cấp, chỉ chọn ngẫu nhiên từ các audio của chữ số đó.

    Args:
        df (pd.DataFrame): DataFrame chứa thông tin audio.
        digit (int, optional): Chữ số cụ thể để chọn audio. 
                               Nếu là None, chọn từ tất cả các chữ số. Mặc định là None.

    Returns:
        IPython.display.Audio: Widget để phát âm thanh, hoặc None nếu không tìm thấy.
    """
    if df.empty:
        print("DataFrame rỗng, không có audio để phát.")
        return None
    

    target_df = df[df['digit'] == digit]
    if target_df.empty:
        print(f"Không tìm thấy audio nào cho chữ số {digit} trong DataFrame.")
        return None


    # Lấy một hàng ngẫu nhiên từ DataFrame (đã được lọc hoặc chưa)
    random_sample = target_df.sample(n=1).iloc[0]
    path = random_sample['path']
    actual_digit = random_sample['digit']
    
    print(f"Phát âm thanh cho chữ số: {actual_digit}")
    
    # Tải và hiển thị sóng âm
    try:
        data, sr = librosa.load(path, sr=None)
        plt.figure(figsize=(10, 3))
        dsp.waveshow(data, sr=sr)
        plt.title(f"Audio of digit: {actual_digit}")
        plt.show()
        
        return Audio(data=data, rate=sr)
    except Exception as e:
        print(f"Lỗi khi tải hoặc hiển thị file {path}: {e}")
        return None
def get_random_audio_raw(df, digit=0):
    """
    Chọn và phát một file audio ngẫu nhiên từ DataFrame.
    Nếu 'digit' được cung cấp, chỉ chọn ngẫu nhiên từ các audio của chữ số đó.

    Args:
        df (pd.DataFrame): DataFrame chứa thông tin audio.
        digit (int, optional): Chữ số cụ thể để chọn audio. 
                               Nếu là None, chọn từ tất cả các chữ số. Mặc định là None.

    Returns:
        IPython.display.Audio: Widget để phát âm thanh, hoặc None nếu không tìm thấy.
    """
    if df.empty:
        print("DataFrame rỗng, không có audio để phát.")
        return None
    

    target_df = df[df['digit'] == digit]
    if target_df.empty:
        print(f"Không tìm thấy audio nào cho chữ số {digit} trong DataFrame.")
        return None


    # Lấy một hàng ngẫu nhiên từ DataFrame (đã được lọc hoặc chưa)
    random_sample = target_df.sample(n=1).iloc[0]
    path = random_sample['path']
    actual_digit = random_sample['digit']
    
    try:
        data, sr = librosa.load(path, sr=None)        
        return Audio(data=data, rate=sr)
    except Exception as e:
        print(f"Lỗi khi tải hoặc hiển thị file {path}: {e}")
        return None
