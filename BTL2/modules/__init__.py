# __init__.py

from .model_eval import (
    preprocess_texts,
    get_embeddings,
    make_clf,
    run_grid_search
)
from .data_loader import(
    load_fake_news_dataset
)
from .EDA import(
    run_eda,
    clean_text,
    preprocess_texts,
    split_dataset
)
from .feature_extraction import(
    extract_features,
    save_features,
    load_features
)