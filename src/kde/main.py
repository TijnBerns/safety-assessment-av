import sys

sys.path.append("src")

from src.kde.parameters import UVParameters as uv_params
import estimator
import click


@click.command()
@click.option("--type", "-t", default="naive_ensemble", type=str)
def main(type: str = "asd"):
    root = uv_params.path_estimates / type
    if type == "combined_data":
        estimator.UnivariatePipeline(estimator.CombinedData).run_pipeline(
            estimator.KDEEstimator(), root
        )
    elif type == "naive_ensemble":
        estimator.UnivariatePipeline(estimator.NaiveEnsemble).run_pipeline(
            estimator.KDEEstimator(), root
        )
        # estimator.MultivariatePipeline(estimator.NaiveEnsemble).run_pipeline(estimator.KDEEstimator(), root)


if __name__ == "__main__":
    main()
