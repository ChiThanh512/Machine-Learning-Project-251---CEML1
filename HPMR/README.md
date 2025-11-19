# Hướng dẫn sử dụng HPMR

## 1. Huấn luyện mô hình trên Google Colab

### Bước 1: Mở notebook huấn luyện
- Mở file `notebooks/main.ipynb` trên Google Colab
- Upload toàn bộ thư mục `HPMR` lên Colab hoặc mount Google Drive

### Bước 2: Cấu hình tham số
Trong notebook, bạn có thể điều chỉnh các tham số của mô hình HMM:
- Số trạng thái ẩn (n_states)
- Số thành phần Gaussian (n_mix)
- Số vòng lặp huấn luyện (n_iter)
- Các tham số MFCC (n_mfcc, n_fft, hop_length)

### Bước 3: Chạy huấn luyện
- Run all cells trong notebook
- Mô hình và features sẽ tự động được lưu vào:
  - Features: `features/processed_data.npz`
  - Model: `models/hmm_model_1/`
  - Metrics: `models/hmm_model_1/metrics.json`

## 2. Test mô hình với âm thanh thực tế

### Bước 1: Mở notebook test
- Mở file `notebooks/test.ipynb` trên Google Colab

### Bước 2: Upload file âm thanh
- Run all cells trong notebook
- Upload file âm thanh định dạng `.wav` khi được yêu cầu
- Mô hình sẽ tự động load từ `models/hmm_model_1/` và dự đoán kết quả

## 3. Cấu trúc thư mục mặc định
