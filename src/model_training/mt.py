import tensorflow as tf
from Mylib import tf_myclasses, tf_myfuncs, myfuncs
import os
import time
from src.utils import classes


def load_data(data_transformation_path):
    train_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "train_ds"))
    val_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "val_ds"))

    return train_ds, val_ds


def create_model_from_layers(model):
    # Này gói input layer trong models luôn, không tách ra như image classfication
    input_layer = model[0]
    x = input_layer

    for layer in model[1:]:
        x = layer(x)

    # Tạo model thôi, chưa cần compile
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    # ✅ Force model to build by passing dummy data
    dummy_input = tf.zeros((1, input_layer.shape[1]), dtype=tf.int32)
    model(dummy_input)

    return model


def create_and_save_models_before_training(
    model_training_path,
    model_indices,
    models,
):
    for model_index, model in zip(model_indices, models):
        model_path = f"{model_training_path}/{model_index}.keras"
        model = create_model_from_layers(model)

        # Save model
        model.save(model_path)


def create_callbacks(
    callbacks,
    model_path,
    best_model_scoring_path,
    target_score,
    model_checkpoint_monitor,
):
    callbacks = [tf_myfuncs.copy_one_callback(callback) for callback in callbacks]

    callbacks = [
        classes.CustomisedModelCheckpoint(
            filepath=model_path,
            scoring_path=best_model_scoring_path,
            monitor=model_checkpoint_monitor,
            indicator=target_score,
        ),
    ] + callbacks

    return callbacks


def get_train_val_scoring_to_plot(train_scoring, val_scoring, scoring):
    if scoring == "accuracy":
        train_scoring, val_scoring = train_scoring * 100, val_scoring * 100

    return train_scoring, val_scoring


def train_and_save_models(
    model_training_path,
    model_indices,
    num_models,
    train_ds,
    val_ds,
    epochs,
    callbacks,
    model_name,
    target_score,
    model_checkpoint_monitor,
    scoring,
    plot_dir,
    optimizer,
    loss,
    metrics,
):
    tf.config.run_functions_eagerly(True)  # Bật eager execution
    tf.data.experimental.enable_debug_mode()  # Bật chế độ eager cho tf.data

    print(
        f"===========Bắt đầu train {num_models} model name = {model_name}===============\n"
    )

    # TODO: d
    print("Run v2")
    # d

    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model
        model_path = os.path.join(model_training_path, f"{model_index}.keras")
        model = tf.keras.models.load_model(model_path)

        # Đường dẫn để lưu scoring ứng với best model
        best_model_scoring_path = os.path.join(
            model_training_path, f"{model_index}_scoring.pkl"
        )

        # Create optimizer cho model
        model_optimizer = tf_myfuncs.copy_one_optimizer(optimizer)

        # Compile model trước khi training
        model.compile(optimizer=model_optimizer, loss=loss, metrics=metrics)

        # Create callbacks cho model
        model_callbacks = create_callbacks(
            callbacks,
            model_path,
            best_model_scoring_path,
            target_score,
            model_checkpoint_monitor,
        )

        # Train model
        print(f"Bắt đầu train model index {model_name} - {model_index}")
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=model_callbacks,
        )
        print(f"Kết thúc train model index {model_name} - {model_index}")

        # TODO: d
        print("Đi kiểm tra đi cha !!!!!!")
        # d

        num_epochs_before_stopping = len(history.history["loss"])

        # Đánh giá model
        train_scoring, val_scoring = myfuncs.load_python_object(best_model_scoring_path)

        ## In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Epochs: {num_epochs_before_stopping}/{epochs}\n"
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        train_scoring, val_scoring = get_train_val_scoring_to_plot(
            train_scoring, val_scoring, scoring
        )

        myfuncs.save_python_object(
            os.path.join(plot_dir, f"{model_name_in_plot}.pkl"),
            (model_name_in_plot, train_scoring, val_scoring),
        )
    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")
    print(
        f"===========Kết thúc train {num_models} model name = {model_name}===============\n"
    )
