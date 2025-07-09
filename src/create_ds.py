from Mylib import tf_myfuncs, myfuncs
from tensorflow.data import Dataset
from tensorflow.keras.layers import TextVectorization


def replace_label_by_number_in_target(target, class_names):
    indices = list(range(len(class_names)))
    class_names_and_indices = list(zip(class_names, indices))
    class_names_and_indices = [([item[0]], item[1]) for item in class_names_and_indices]
    target_replaced = myfuncs.replace_in_series(target, class_names_and_indices)
    return target_replaced


def create_train_val_ds(param, train_val_path, class_names):
    print("Tạo ds cho ngram model")

    train_feature = myfuncs.load_python_object(train_val_path / "train_feature.pkl")
    train_target = myfuncs.load_python_object(train_val_path / "train_target.pkl")
    train_target = replace_label_by_number_in_target(train_target, class_names)

    val_feature = myfuncs.load_python_object(train_val_path / "val_feature.pkl")
    val_target = myfuncs.load_python_object(train_val_path / "val_target.pkl")
    val_target = replace_label_by_number_in_target(val_target, class_names)

    train_ds = Dataset.from_tensor_slices((train_feature, train_target))
    train_ds = train_ds.batch(param["batch_size"])
    val_ds = Dataset.from_tensor_slices((val_feature, val_target))
    val_ds = val_ds.batch(param["batch_size"])

    text_vectorization = TextVectorization(
        ngrams=param["ngrams"],
        max_tokens=param["max_tokens"],
        output_mode=param["output_mode"],
    )

    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    train_ds_transformed = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )
    val_ds_transformed = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )

    train_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_no_shuffle(
        train_ds_transformed
    )
    val_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_no_shuffle(
        val_ds_transformed
    )

    return train_ds_transformed, val_ds_transformed


def create_train_val_ds_for_sequence_model(param, train_val_path, class_names):
    print("Tạo ds cho sequence model")

    train_feature = myfuncs.load_python_object(train_val_path / "train_feature.pkl")
    train_target = myfuncs.load_python_object(train_val_path / "train_target.pkl")
    train_target = replace_label_by_number_in_target(train_target, class_names)

    val_feature = myfuncs.load_python_object(train_val_path / "val_feature.pkl")
    val_target = myfuncs.load_python_object(train_val_path / "val_target.pkl")
    val_target = replace_label_by_number_in_target(val_target, class_names)

    train_ds = Dataset.from_tensor_slices((train_feature, train_target))
    train_ds = train_ds.batch(param["batch_size"])
    val_ds = Dataset.from_tensor_slices((val_feature, val_target))
    val_ds = val_ds.batch(param["batch_size"])

    text_vectorization = TextVectorization(
        max_tokens=param["max_tokens"],
        output_mode="int",
        output_sequence_length=param["output_sequence_length"],
    )

    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    train_ds_transformed = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )
    val_ds_transformed = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
    )

    train_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_no_shuffle(
        train_ds_transformed
    )
    val_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_no_shuffle(
        val_ds_transformed
    )

    return train_ds_transformed, val_ds_transformed
