from kedro.pipeline import Pipeline, node, pipeline

from asi_project.pipelines.data_processing.nodes import (
    input_normalization,
    training_only_filtering,
    validate_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data,
                inputs="raw_data",
                outputs="validated_data",
                name="validate_data_node",
            ),
            node(
                func=input_normalization,
                inputs="validated_data",
                outputs="normalized_data",
                name="input_normalization_node",
            ),
            node(
                func=training_only_filtering,
                inputs="normalized_data",
                outputs="model_input_table",
                name="training_only_filtering_node",
            ),
        ]
    )
