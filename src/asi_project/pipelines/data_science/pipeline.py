from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_model_comparison,
    compute_data_fingerprint,
    evaluate_model,
    log_baseline_run,
    split_data,
    train_autogluon_experiments,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compute_data_fingerprint,
                inputs="params:raw_data_path",
                outputs="data_fingerprint",
                name="compute_data_fingerprint_node",
            ),
            node(
                func=split_data,
                inputs=["model_input_table", "params:split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:numeric_features",
                    "params:categorical_features",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
            node(
                func=log_baseline_run,
                inputs=[
                    "trained_model",
                    "metrics",
                    "X_test",
                    "params:split_params",
                    "data_fingerprint",
                    "params:wandb",
                    "params:comparison",
                ],
                outputs="baseline_run",
                name="log_baseline_run_node",
            ),
            node(
                func=train_autogluon_experiments,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:autogluon",
                    "params:split_params",
                    "data_fingerprint",
                    "params:wandb",
                    "baseline_run",
                    "params:comparison",
                ],
                outputs="autogluon_metrics",
                name="train_autogluon_node",
            ),
            node(
                func=build_model_comparison,
                inputs=[
                    "baseline_run",
                    "autogluon_metrics",
                    "params:split_params",
                    "data_fingerprint",
                    "params:wandb",
                ],
                outputs=["model_comparison", "model_comparison_md"],
                name="build_model_comparison_node",
            ),
        ]
    )
