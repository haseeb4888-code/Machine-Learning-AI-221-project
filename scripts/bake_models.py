from __future__ import annotations

from pathlib import Path

from src.models.training import TrainingPipeline
from src.models.classification_models import ClassificationModelManager
from src.models.regression_models import RegressionModelManager


def find_dataset_path() -> Path | None:
    candidates = [
        Path("Old-Version") / "countries of the world.csv",
        Path("old-Version") / "countries of the world.csv",
    ]
    print("Dataset candidates:", [str(p) for p in candidates])
    return next((p for p in candidates if p.exists()), None)


def main() -> int:
    data_path = find_dataset_path()
    if data_path is None:
        print("! Skipping model baking (CSV not found in build context)")
        return 0

    pipeline = TrainingPipeline(str(data_path))
    pipeline.load_and_preprocess()
    pipeline.engineer_features()
    pipeline.create_target_variable()
    pipeline.prepare_train_test_split()

    clf = ClassificationModelManager()
    clf.train_logistic_regression(
        pipeline.X_train, pipeline.y_train_clf, pipeline.X_test, pipeline.y_test_clf
    )
    clf.train_random_forest(
        pipeline.X_train, pipeline.y_train_clf, pipeline.X_test, pipeline.y_test_clf
    )
    clf.save_models()

    reg = RegressionModelManager()
    reg.train_linear_regression(
        pipeline.X_train, pipeline.y_train_reg, pipeline.X_test, pipeline.y_test_reg
    )
    reg.train_random_forest_regressor(
        pipeline.X_train, pipeline.y_train_reg, pipeline.X_test, pipeline.y_test_reg
    )
    reg.train_gradient_boosting(
        pipeline.X_train, pipeline.y_train_reg, pipeline.X_test, pipeline.y_test_reg
    )
    reg.save_models()

    print("\u2713 Baked models into image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
