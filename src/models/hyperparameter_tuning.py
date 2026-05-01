"""Hyperparameter tuning using Optuna.

Implements sequential (non-parallel) hyperparameter search for selected
classification + regression models in this project.

Tuning happens BEFORE final training:
- Optuna tests n_trials candidate hyperparameter sets via CV
- Best params are returned and later used for final model training
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 50
    timeout: int = 300  # seconds
    seed: int = 42
    cv: int = 5


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna."""

    def __init__(self, n_trials: int = 50, timeout: int = 300, seed: int = 42):
        self.config = TuningConfig(n_trials=n_trials, timeout=timeout, seed=seed)
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.study_history: Dict[str, optuna.study.Study] = {}

    def _run_study(
        self,
        objective: Callable[[optuna.Trial], float],
        *,
        direction: str,
        study_name: str,
    ) -> optuna.study.Study:
        sampler = TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=False,
            n_jobs=1,  # sequential search (required by the prompt)
        )
        return study

    # =========================================================================
    # CLASSIFICATION TUNING
    # =========================================================================

    def tune_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune Logistic Regression classifier hyperparameters."""
        print("\n🔍 Tuning Logistic Regression...")

        def objective(trial: optuna.Trial) -> float:
            c = trial.suggest_float("C", 1e-3, 100.0, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 2000)

            # Keep solver compatible with L2 penalty; avoids solver/penalty mismatches.
            model = LogisticRegression(
                C=c,
                max_iter=max_iter,
                random_state=self.config.seed,
                solver="lbfgs",
                penalty="l2",
            )

            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring="accuracy", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="logistic_regression")
        self.best_params["logistic_regression"] = study.best_params
        self.study_history["logistic_regression"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    def tune_random_forest_classifier(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Tune RandomForestClassifier hyperparameters."""
        print("\n🔍 Tuning Random Forest Classifier...")

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 5, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.config.seed,
                n_jobs=-1,
            )

            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring="accuracy", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="random_forest_clf")
        self.best_params["random_forest_clf"] = study.best_params
        self.study_history["random_forest_clf"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    def tune_xgboost_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune XGBClassifier hyperparameters."""
        print("\n🔍 Tuning XGBoost Classifier...")

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=self.config.seed,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=1,  # keep sequential tuning predictable; we still allow RF internal n_jobs=-1 above
            )

            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring="accuracy", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="xgboost_clf")
        self.best_params["xgboost_clf"] = study.best_params
        self.study_history["xgboost_clf"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    def tune_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune SVM (Support Vector Machine) classifier hyperparameters."""
        print("\n🔍 Tuning SVM Classifier...")

        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
            
            if kernel == "poly":
                degree = trial.suggest_int("degree", 2, 5)
            else:
                degree = 3  # Default, not used for other kernels

            # Scale features for SVM (important for performance)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                random_state=self.config.seed,
                probability=True,  # Enable probability estimates
            )

            scores = cross_val_score(model, X_scaled, y_train, cv=self.config.cv, scoring="accuracy", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="svm_clf")
        self.best_params["svm_clf"] = study.best_params
        self.study_history["svm_clf"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    # =========================================================================
    # REGRESSION TUNING
    # =========================================================================

    def tune_random_forest_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune RandomForestRegressor hyperparameters."""
        print("\n🔍 Tuning Random Forest Regressor...")

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 5, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.config.seed,
                n_jobs=-1,
            )

            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring="r2", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="random_forest_reg")
        self.best_params["random_forest_reg"] = study.best_params
        self.study_history["random_forest_reg"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    def tune_xgboost_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune XGBRegressor hyperparameters."""
        print("\n🔍 Tuning XGBoost Regressor...")

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=self.config.seed,
                objective="reg:squarederror",
                n_jobs=1,
            )

            scores = cross_val_score(model, X_train, y_train, cv=self.config.cv, scoring="r2", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="xgboost_reg")
        self.best_params["xgboost_reg"] = study.best_params
        self.study_history["xgboost_reg"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    def tune_svm_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune SVM (Support Vector Machine) regressor hyperparameters."""
        print("\n🔍 Tuning SVM Regressor...")

        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 0.01, 0.5, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
            
            if kernel == "poly":
                degree = trial.suggest_int("degree", 2, 5)
            else:
                degree = 3  # Default, not used for other kernels

            # Scale features for SVM (important for performance)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = SVR(
                C=C,
                epsilon=epsilon,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
            )

            scores = cross_val_score(model, X_scaled, y_train, cv=self.config.cv, scoring="r2", n_jobs=1)
            return float(np.mean(scores))

        study = self._run_study(objective, direction="maximize", study_name="svm_reg")
        self.best_params["svm_reg"] = study.best_params
        self.study_history["svm_reg"] = study

        print(f"✓ Best params: {study.best_params}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        return study.best_params

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """Get best parameters for a specific tuned model."""
        return self.best_params.get(model_name, {})

    def get_all_best_params(self) -> Dict[str, Dict[str, Any]]:
        """Get all best parameters."""
        return self.best_params

    def save_best_params(self, filepath: str | Path = "hyperparameters.json") -> Path:
        """Save best parameters to a JSON file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)
        print(f"✓ Best parameters saved to {out_path}")
        return out_path

    def print_summary(self) -> None:
        """Print summary of tuning results."""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("=" * 60)

        if not self.best_params:
            print("No tuning results found.")
            return

        for model_name, params in self.best_params.items():
            print(f"\n{model_name}:")
            for param_name, param_value in params.items():
                print(f"  {param_name}: {param_value}")

        print("\n" + "=" * 60)
