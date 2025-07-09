import tensorflow as tf
from Mylib import tf_myfuncs, myfuncs
import os
from src.utils import funcs


def load_data(data_ingestion_path):
    train_feature = myfuncs.load_python_object(
        f"{data_ingestion_path}/train_feature.pkl"
    )
    train_target = myfuncs.load_python_object(f"{data_ingestion_path}/train_target.pkl")
    val_feature = myfuncs.load_python_object(f"{data_ingestion_path}/val_feature.pkl")
    val_target = myfuncs.load_python_object(f"{data_ingestion_path}/val_target.pkl")

    return train_feature, train_target, val_feature, val_target


def create_vocabulary_transform_and_save_data(
    train_feature,
    train_target,
    val_feature,
    val_target,
    batch_size,
    text_vectorization,
    data_transformation_path,
):
    # Tạo train_ds và val_ds
    train_ds = tf.data.Dataset.from_tensor_slices((train_feature, train_target))
    train_ds = train_ds.batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_feature, val_target))
    val_ds = val_ds.batch(batch_size)

    # Tạo vocabulary
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    # Transform data
    train_ds_transformed = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )
    val_ds_transformed = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )

    # Cache data
    train_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_2(train_ds_transformed)
    val_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_2(val_ds_transformed)

    # Save data
    train_ds_transformed.save(os.path.join(data_transformation_path, "train_ds"))
    val_ds_transformed.save(os.path.join(data_transformation_path, "val_ds"))
    tf_myfuncs.save_text_vectorization(
        text_vectorization,
        os.path.join(data_transformation_path, "text_vectorization.keras"),
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/vocab_size.pkl", text_vectorization.max_tokens
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/output_sequence_length.pkl",
        text_vectorization.output_sequence_length,
    )
