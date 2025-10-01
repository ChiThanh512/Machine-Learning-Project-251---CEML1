
def download_kaggle_dataset(dataset_name, username, key, download_dir='./data'):
    """
    Tải và giải nén một dataset từ Kaggle vào một thư mục cụ thể.

    Args:
        dataset_name (str): Tên của dataset trên Kaggle (ví dụ: 'user/dataset-name').
        username (str): Tên người dùng Kaggle.
        key (str): Khóa API Kaggle.
        download_dir (str): Thư mục để lưu và giải nén dataset.
    """
    print("--- Bắt đầu quá trình tải dataset từ Kaggle ---")

    # 1. Cài đặt thư viện Kaggle một cách yên lặng
    print("1. Cài đặt thư viện Kaggle...")
    !pip install kaggle --quiet

    # 2. Thiết lập thông tin xác thực Kaggle
    print("2. Thiết lập thông tin xác thực...")
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    # 3. Tạo thư mục đích
    print(f"3. Đảm bảo thư mục '{download_dir}' tồn tại...")
    os.makedirs(download_dir, exist_ok=True)

    # 4. Tải dataset
    # Lệnh này sẽ tải file zip vào thư mục download_dir
    print(f"4. Tải dataset '{dataset_name}'...")
    !kaggle datasets download -d {dataset_name} -p {download_dir} --force

    # 5. Giải nén file zip
    # Lấy tên file zip từ tên dataset
    zip_file_name = dataset_name.split('/')[-1] + '.zip'
    zip_file_path = os.path.join(download_dir, zip_file_name)
    
    print(f"5. Giải nén file '{zip_file_path}'...")
    if os.path.exists(zip_file_path):
        !unzip -o -q {zip_file_path} -d {download_dir}
        print("   Giải nén thành công.")
    else:
        print(f"   Lỗi: Không tìm thấy file zip '{zip_file_path}' để giải nén.")

    # 6. Hoàn tất
    print("\n--- Quá trình hoàn tất! ---")
    print(f"Dữ liệu đã sẵn sàng trong thư mục '{download_dir}'.")
    !ls {download_dir}