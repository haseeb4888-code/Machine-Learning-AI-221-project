"""DeepChecks-based validation tests.

These tests verify that the DeepChecks suites can run end-to-end on
representative (synthetic) data. This gives you a CI gate for data/model
sanity beyond unit tests.
"""

import pytest

deepchecks = pytest.importorskip("deepchecks")

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation, model_evaluation



def test_deepchecks_train_test_validation_runs():
    """Runs DeepChecks train/test validation suite without errors."""
    X, y = make_classification(
        n_samples=400,
        n_features=18,
        n_informative=12,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    train_ds = Dataset(
        pd.DataFrame(X_train, columns=feature_names),
        label=pd.Series(y_train, name="label"),
    )
    test_ds = Dataset(
        pd.DataFrame(X_test, columns=feature_names),
        label=pd.Series(y_test, name="label"),
    )

    suite = train_test_validation()
    result = suite.run(train_dataset=train_ds, test_dataset=test_ds)

    # The main goal is that the suite executes successfully in CI.
    assert result is not None


def test_deepchecks_model_evaluation_runs_classification():
    """Runs DeepChecks model evaluation suite for a classifier."""
    X, y = make_classification(
        n_samples=1200,
        n_features=18,
        n_informative=12,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    train_ds = Dataset(
        pd.DataFrame(X_train, columns=feature_names),
        label=pd.Series(y_train, name="label"),
    )
    test_ds = Dataset(
        pd.DataFrame(X_test, columns=feature_names),
        label=pd.Series(y_test, name="label"),
    )

    suite = model_evaluation()
    result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
    assert result is not None


def test_deepchecks_model_evaluation_runs_regression():
    """Runs DeepChecks model evaluation suite for a regressor."""
    X, y = make_regression(
        n_samples=400,
        n_features=18,
        n_informative=12,
        noise=0.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    train_ds = Dataset(
        pd.DataFrame(X_train, columns=feature_names),
        label=pd.Series(y_train, name="target"),
    )
    test_ds = Dataset(
        pd.DataFrame(X_test, columns=feature_names),
        label=pd.Series(y_test, name="target"),
    )

    suite = model_evaluation()
    result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
    assert result is not None
