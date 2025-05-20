import tensorflow as tf
from Mylib import tf_myfuncs
import os
from src.utils import funcs


def load_data(train_ds_path, val_ds_path):
    train_ds = tf.data.Dataset.load(train_ds_path)
    val_ds = tf.data.Dataset.load(val_ds_path)

    return train_ds, val_ds


def create_vocabulary_transform_and_save_data(
    train_ds, val_ds, text_vectorization, data_transformation_path
):
    # Tạo vocabulary
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    # Transform data
    train_ds_transformed = funcs.transform_data_with_text_vectorization(
        train_ds, text_vectorization
    )
    val_ds_transformed = funcs.transform_data_with_text_vectorization(
        val_ds, text_vectorization
    )

    # Save data
    train_ds_transformed.save(os.path.join(data_transformation_path, "train_ds"))
    val_ds_transformed.save(os.path.join(data_transformation_path, "val_ds"))

    # Save text_vectorization để sau này dùng lại
    funcs.save_text_vectorization(
        text_vectorization,
        os.path.join(data_transformation_path, "text_vectorization.keras"),
    )
