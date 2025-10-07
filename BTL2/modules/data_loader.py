# ============================================================
# modules/data_loader.py
# ============================================================
import pandas as pd
import os
import zipfile
from IPython.display import display

def load_fake_news_dataset(repo_url="https://github.com/NhutTomorrow/Fake-News-dataset.git"):
    """
    Clone dataset, extract files, clean data, and return:
    - df: full merged dataframe (True + Fake)
    - df_manual_testing: 20 manually separated samples (10 Fake + 10 True)
    """

    # --- Clone repo nếu chưa có ---
    if not os.path.exists("Fake-News-dataset"):
        os.system(f"git clone {repo_url}")

    # --- Giải nén ---
    os.makedirs("Fake-News-dataset/Fake_csv", exist_ok=True)
    os.makedirs("Fake-News-dataset/True_csv", exist_ok=True)

    os.system("unzip -o Fake-News-dataset/Fake.csv.zip -d Fake-News-dataset/Fake_csv")
    os.system("unzip -o Fake-News-dataset/True.csv.zip -d Fake-News-dataset/True_csv")

    # --- Đọc file ---
    fake_df = pd.read_csv("Fake-News-dataset/Fake_csv/Fake.csv")
    true_df = pd.read_csv("Fake-News-dataset/True_csv/True.csv")

    print("✅ True dataset:", true_df.shape)  
    print("✅ Fake dataset:", fake_df.shape)

    # --- Thêm nhãn ---
    true_df["label"] = 1   # Tin thật
    fake_df["label"] = 0   # Tin giả

    # --- Chuẩn bị manual testing ---
    df_fake_manual_testing = fake_df.tail(10).copy()
    fake_df = fake_df.iloc[:-10, :]

    df_true_manual_testing = true_df.tail(10).copy()
    true_df = true_df.iloc[:-10, :]

    df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
    df_manual_testing.to_csv("manual_testing.csv", index=False)

    # --- Kết hợp dữ liệu chính ---
    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    print("✅ Combined dataset:", df.shape)
    return df, df_manual_testing
