import argparse
import os
import time
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import optuna
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from optuna.trial import TrialState
from xgboost import XGBClassifier


SCRIPT_START_TIME = time.perf_counter()
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.03, "log": True},
    "max_depth": {"type": "int", "low": 4, "high": 10},
    "min_child_weight": {"type": "int", "low": 1, "high": 12},
    "max_delta_step": {"type": "int", "low": 0, "high": 3},
    "gamma": {"type": "float", "low": 0.0, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 2.0},
    "reg_lambda": {"type": "float", "low": 0.5, "high": 6.0},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bylevel": {"type": "float", "low": 0.6, "high": 1.0},
}


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Could not find repo root containing .git")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bands = ["u", "g", "r", "i", "z"]

    df["mag_mean"] = df[bands].mean(axis=1)
    df["mag_std"] = df[bands].std(axis=1)
    df["mag_max"] = df[bands].max(axis=1)
    df["mag_min"] = df[bands].min(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]

    alpha_rad = np.radians(df["alpha"])
    delta_rad = np.radians(df["delta"])
    df["alpha_sin"] = np.sin(alpha_rad)
    df["alpha_cos"] = np.cos(alpha_rad)
    df["delta_sin"] = np.sin(delta_rad)
    df["delta_cos"] = np.cos(delta_rad)

    epsilon = 1e-5
    abs_redshift = df["redshift"].abs() + epsilon
    for col in ["u_g", "g_r", "r_i", "i_z"]:
        df[f"{col}_per_redshift"] = df[col] / abs_redshift
    for band in bands:
        df[f"redshift_{band}"] = df["redshift"] * df[band]

    return df


def prepare_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", index_col="id")

    X = add_features(train).drop(columns=["class"])
    categorical_cols = ["spectral_type", "galaxy_population"]
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train["class"])

    return X, y


def prompt_yes_no(question: str) -> bool:
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def make_storage(storage_url: str) -> optuna.storages.RDBStorage:
    return optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=60,
        grace_period=300,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(
            max_retry=1
        ),
    )


def study_exists(study_name: str, storage: optuna.storages.RDBStorage) -> bool:
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    return any(summary.study_name == study_name for summary in summaries)


def get_resume_attr(study: optuna.Study, key: str, fallback):
    if key in study.user_attrs:
        return study.user_attrs[key]
    print(f"Study is missing saved '{key}' metadata; using fallback value: {fallback}")
    return fallback


def suggest_float_from_space(trial: optuna.Trial, name: str) -> float:
    spec = SEARCH_SPACE[name]
    return trial.suggest_float(
        name,
        spec["low"],
        spec["high"],
        log=spec.get("log", False),
    )


def suggest_int_from_space(trial: optuna.Trial, name: str) -> int:
    spec = SEARCH_SPACE[name]
    return trial.suggest_int(name, spec["low"], spec["high"])


def retry_running_trials(study: optuna.Study) -> None:
    optuna.storages.fail_stale_trials(study)
    running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    if not running_trials:
        return

    print(f"Found {len(running_trials)} RUNNING trial(s) in this study.")
    print(
        "These are usually interrupted trials if no other training process is still active."
    )
    if not prompt_yes_no("Mark them failed and requeue their parameter sets?"):
        raise SystemExit("Canceled resume because RUNNING trials were left untouched.")

    for trial in running_trials:
        if trial.params:
            study.enqueue_trial(
                trial.params,
                user_attrs={"retry_of_interrupted_trial": trial.number},
                skip_if_exists=False,
            )
        study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
        print(f"Requeued interrupted trial {trial.number}.")


def enqueue_best_params_from_study(
    *,
    target_study: optuna.Study,
    source_study_name: str,
    storage: optuna.storages.RDBStorage,
) -> None:
    if not study_exists(source_study_name, storage):
        raise SystemExit(f"Cannot seed from '{source_study_name}' because it does not exist.")

    source_study = optuna.load_study(study_name=source_study_name, storage=storage)
    best_params = source_study.best_params
    target_study.enqueue_trial(
        best_params,
        user_attrs={
            "seeded_from_study": source_study_name,
            "seeded_from_trial": source_study.best_trial.number,
            "seeded_from_value": source_study.best_value,
        },
        skip_if_exists=True,
    )
    print(
        f"Queued best params from study '{source_study_name}' "
        f"(trial {source_study.best_trial.number}, value {source_study.best_value:.12f})."
    )


def make_objective(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    n_estimators: int,
    early_stopping_rounds: int,
    device: str,
):
    def objective(trial: optuna.Trial) -> float:
        trial_start_time = time.perf_counter()
        print()
        print(f"Running trial {trial.number}")

        params = {
            "learning_rate": suggest_float_from_space(trial, "learning_rate"),
            "max_depth": suggest_int_from_space(trial, "max_depth"),
            "min_child_weight": suggest_int_from_space(trial, "min_child_weight"),
            "max_delta_step": suggest_int_from_space(trial, "max_delta_step"),
            "gamma": suggest_float_from_space(trial, "gamma"),
            "reg_alpha": suggest_float_from_space(trial, "reg_alpha"),
            "reg_lambda": suggest_float_from_space(trial, "reg_lambda"),
            "subsample": suggest_float_from_space(trial, "subsample"),
            "colsample_bytree": suggest_float_from_space(trial, "colsample_bytree"),
            "colsample_bylevel": suggest_float_from_space(trial, "colsample_bylevel"),
        }
        print("Trial params:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6g}")
            else:
                print(f"  {key}: {value}")
        print(f"  n_estimators: {n_estimators}")
        print(f"  early_stopping_rounds: {early_stopping_rounds}")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        oof_probs = np.zeros((len(X), 3))
        fold_stats = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            fold_start_time = time.perf_counter()
            print(f"Training fold {fold}/{n_splits}")

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                device=device,
                tree_method="hist",
                random_state=random_state,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                enable_categorical=True,
                eval_metric="mlogloss",
                verbosity=0,
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            oof_probs[val_idx] = model.predict_proba(X_val)
            evals_result = model.evals_result()
            validation_metric = evals_result["validation_0"]["mlogloss"]
            rounds_trained = len(validation_metric)
            best_iteration = int(model.best_iteration)
            best_score = float(model.best_score)
            stopped_by_early_stopping = rounds_trained < n_estimators
            best_iteration_gap_to_cap = n_estimators - (best_iteration + 1)
            rounds_after_best = rounds_trained - (best_iteration + 1)
            fold_stats.append(
                {
                    "fold": fold,
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                    "rounds_trained": rounds_trained,
                    "rounds_after_best": rounds_after_best,
                    "best_iteration_gap_to_cap": best_iteration_gap_to_cap,
                    "stopped_by_early_stopping": stopped_by_early_stopping,
                }
            )
            fold_elapsed = time.perf_counter() - fold_start_time
            print(
                f"Fold {fold}/{n_splits} completed in {format_duration(fold_elapsed)} "
                f"(best_iteration={best_iteration}, rounds_trained={rounds_trained})"
            )

        oof_preds = np.argmax(oof_probs, axis=1)
        score = balanced_accuracy_score(y, oof_preds)
        best_iterations = [item["best_iteration"] for item in fold_stats]
        rounds_trained_values = [item["rounds_trained"] for item in fold_stats]
        trial.set_user_attr("fold_stats", fold_stats)
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iterations)))
        trial.set_user_attr("max_best_iteration", int(np.max(best_iterations)))
        trial.set_user_attr("mean_rounds_trained", float(np.mean(rounds_trained_values)))
        trial.set_user_attr("max_rounds_trained", int(np.max(rounds_trained_values)))
        trial.set_user_attr(
            "hit_n_estimators_cap",
            bool(any(rounds == n_estimators for rounds in rounds_trained_values)),
        )
        trial_elapsed = time.perf_counter() - trial_start_time
        total_elapsed = time.perf_counter() - SCRIPT_START_TIME
        print(f"Trial {trial.number} OOF balanced accuracy: {score:.12f}")
        print(
            "Best iteration range: "
            f"{min(best_iterations)}-{max(best_iterations)} "
            f"(mean {np.mean(best_iterations):.1f})"
        )
        print(f"Trial {trial.number} duration: {format_duration(trial_elapsed)}")
        print(f"Total training time so far: {format_duration(total_elapsed)}")
        return score

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run persistent Optuna tuning for the stellar class XGBoost model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Start a fresh study. If the name exists, ask before overwriting it.",
    )
    train_parser.add_argument("study_name", help="Optuna study name.")
    train_parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    train_parser.add_argument(
        "--seed-from-study",
        default=None,
        help=(
            "Queue the best parameter set from an existing study as the first trial "
            "of this new study. Omit this for a clean unseeded search."
        ),
    )
    train_parser.add_argument("--n-splits", type=int, default=5, help="Number of StratifiedKFold splits.")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random state for folds and XGBoost.")
    train_parser.add_argument("--n-estimators", type=int, default=6000, help="Maximum XGBoost boosting rounds.")
    train_parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=150,
        help="XGBoost early stopping patience.",
    )
    train_parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI. Defaults to sqlite:///<repo>/Models/XGBoost/optuna_xgboost.db.",
    )
    train_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. auto uses CUDA when torch can see it.",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing study and reuse its saved run metadata.",
    )
    resume_parser.add_argument("study_name", help="Optuna study name to resume.")
    resume_parser.add_argument("--n-trials", type=int, default=10, help="Number of additional Optuna trials to run.")
    resume_parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI. Defaults to sqlite:///<repo>/Models/XGBoost/optuna_xgboost.db.",
    )
    resume_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Fallback only for older studies missing saved device metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "/usr/lib/wsl/lib" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("PATH", "")
    load_dotenv()

    root = find_repo_root(Path.cwd())
    data_dir = root / "Data"
    complete_file = data_dir / ".complete/competitions/playground-series-s6e6/bundle.complete"
    if not complete_file.is_file():
        kagglehub.competition_download("playground-series-s6e6", output_dir=data_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    storage_url = args.storage
    if storage_url is None:
        storage_url = f"sqlite:///{root / 'Models/XGBoost/optuna_xgboost.db'}"
    storage = make_storage(storage_url)

    exists = study_exists(args.study_name, storage)
    is_resume = args.command == "resume"
    if is_resume:
        if not exists:
            raise SystemExit(
                f"Cannot resume study '{args.study_name}' because it does not exist."
            )
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        args.n_splits = int(get_resume_attr(study, "n_splits", 5))
        args.random_state = int(get_resume_attr(study, "random_state", 42))
        args.n_estimators = int(
            get_resume_attr(study, "n_estimators", 6000)
        )
        args.early_stopping_rounds = int(
            get_resume_attr(
                study,
                "early_stopping_rounds",
                150,
            )
        )
        device = str(get_resume_attr(study, "device", device))
        retry_running_trials(study)
    else:
        if exists:
            print(f"Study '{args.study_name}' already exists.")
            if not prompt_yes_no("Overwrite it and start fresh?"):
                raise SystemExit(
                    f"Canceled. Use `resume {args.study_name}` to continue the existing study."
                )
            optuna.delete_study(study_name=args.study_name, storage=storage)
            print(f"Deleted existing study '{args.study_name}'.")
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=False,
        )
        study.set_user_attr("n_estimators", args.n_estimators)
        study.set_user_attr("early_stopping_rounds", args.early_stopping_rounds)
        study.set_user_attr("n_splits", args.n_splits)
        study.set_user_attr("random_state", args.random_state)
        study.set_user_attr("device", device)
        study.set_user_attr("data_backend", "pandas")
        study.set_user_attr("categorical_encoding", "native pandas category")
        study.set_user_attr("search_space", SEARCH_SPACE)
        study.set_user_attr("seed_from_study", args.seed_from_study)
        if args.seed_from_study is not None:
            enqueue_best_params_from_study(
                target_study=study,
                source_study_name=args.seed_from_study,
                storage=storage,
            )

    X, y = prepare_data(data_dir)
    print(f"Rows: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Repo root: {root}")
    print(f"Data dir: {data_dir}")
    print(f"Study name: {args.study_name}")
    print(f"Mode: {args.command}")
    print(f"Storage: {storage_url}")
    print(f"Device: {device}")
    print(f"Trials to run: {args.n_trials}")
    print(f"CV splits: {args.n_splits}")
    print(f"Random state: {args.random_state}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"early_stopping_rounds: {args.early_stopping_rounds}")
    print("Data backend: pandas")
    print("Categorical encoding: native pandas category")

    objective = make_objective(
        X,
        y,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        device=device,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print()
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
