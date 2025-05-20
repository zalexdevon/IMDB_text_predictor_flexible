import tensorflow as tf
from Mylib import myfuncs, tf_myclasses
import os
from utils import funcs


def load_data(test_ds_path, data_transformation_path, class_names_path, model_path):
    test_ds = tf.data.Dataset.load(test_ds_path)
    text_vectorization = funcs.load_text_vectorization(
        os.path.join(data_transformation_path, "text_vectorization.keras")
    )
    class_names = myfuncs.load_python_object(class_names_path)
    model = tf.keras.models.load_model(model_path)

    return test_ds, text_vectorization, class_names, model


def evaluate_model_on_test(test_ds, class_names, model, model_evaluation_on_test_path):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, test_confusion_matrix = tf_myclasses.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_ds=test_ds,
    ).evaluate()
    final_model_results_text += model_results_text  # Thêm đoạn đánh giá vào

    # Lưu lại confusion matrix cho tập train và val
    test_confusion_matrix_path = os.path.join(
        model_evaluation_on_test_path, "test_confusion_matrix.png"
    )
    test_confusion_matrix.savefig(
        test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(
        os.path.join(model_evaluation_on_test_path, "result.txt"), mode="w"
    ) as file:
        file.write(final_model_results_text)
