import os

def download_kaggle_dataset(dataset_name, username, key, download_dir='./data'):
    """
    Tải và giải nén một dataset từ Kaggle vào một thư mục cụ thể.
    Hàm này sử dụng os.system() để chạy các lệnh shell.
    """
    print("--- Bắt đầu quá trình tải dataset từ Kaggle ---")

    # 1. Cài đặt thư viện Kaggle một cách yên lặng
    print("1. Cài đặt thư viện Kaggle...")
    os.system('pip install kaggle --quiet')

    # 2. Thiết lập thông tin xác thực Kaggle
    print("2. Thiết lập thông tin xác thực...")
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    # 3. Tạo thư mục đích
    print(f"3. Đảm bảo thư mục '{download_dir}' tồn tại...")
    os.makedirs(download_dir, exist_ok=True)

    # 4. Tải dataset bằng lệnh shell
    print(f"4. Tải dataset '{dataset_name}'...")
    download_command = f"kaggle datasets download -d {dataset_name} -p {download_dir} --force"
    os.system(download_command)

    # 5. Giải nén file zip
    zip_file_name = dataset_name.split('/')[-1] + '.zip'
    zip_file_path = os.path.join(download_dir, zip_file_name)
    
    print(f"5. Giải nén file '{zip_file_path}'...")
    if os.path.exists(zip_file_path):
        unzip_command = f"unzip -o -q {zip_file_path} -d {download_dir}"
        os.system(unzip_command)
        print("   Giải nén thành công.")
    else:
        print(f"   Lỗi: Không tìm thấy file zip '{zip_file_path}' để giải nén.")

    # 6. Hoàn tất
    print("\n--- Quá trình hoàn tất! ---")
    print(f"Dữ liệu đã sẵn sàng trong thư mục '{download_dir}'.")
    os.system(f'ls {download_dir}')