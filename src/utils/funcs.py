import tensorflow as tf
from Mylib import tf_myfuncs


def save_text_vectorization(text_vectorization, file_path):
    vectorizer_model = tf.keras.Sequential([text_vectorization])
    vectorizer_model.save(file_path)


def load_text_vectorization(file_path):
    loaded_model = tf.keras.models.load_model(file_path)
    return loaded_model.layers[
        0
    ]  # Lúc save thì cho vào dummy model, nên lúc lấy ra thì get layer đầu tiên


def transform_data_with_text_vectorization(ds, text_vectorization):
    ds_transformed = ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )

    # Cache và prefetch
    ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_2(ds_transformed)

    return ds_transformed
