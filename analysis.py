
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler,
)

SEED = 42
BASE = ["Pclass", "SibSp", "Parch", "Fare"]
NUM = ["Age", "SibSp", "Parch", "Fare"]
CAT = ["Pclass", "Sex", "Embarked"]
DERIVED = ["FamilySize", "FarePerPerson"]
REQUIRED = {"PassengerId", "Survived", "Ticket", "Cabin", *NUM, *CAT}
CORE = ("Base", "Exp-1", "Exp-2", "Exp-3")
CONFIGS = {
    "Exp-1": ("mean", "One-Hot", "Standard", True, False),
    "Exp-2": ("median", "Label", "MinMax", True, True),
    "Exp-3": ("most_frequent", "One-Hot", "Robust", True, True),
    "A-Median": ("median", "One-Hot", "Standard", True, False),
    "A-Mode": ("most_frequent", "One-Hot", "Standard", True, False),
    "A-Label": ("mean", "Label", "Standard", True, False),
    "A-MinMax": ("mean", "One-Hot", "MinMax", True, False),
    "A-Robust": ("mean", "One-Hot", "Robust", True, False),
    "A-NoDerived": ("mean", "One-Hot", "Standard", False, False),
    "A-Top6": ("mean", "One-Hot", "Standard", True, True),
    "A-Exp2-All": ("median", "Label", "MinMax", True, False),
    "A-Exp3-All": ("most_frequent", "One-Hot", "Robust", True, False),
}
SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
MODELS = ("LR", "RF")


def add_features(frame):
    size = frame["SibSp"] + frame["Parch"] + 1
    return frame.assign(FamilySize=size, FarePerPerson=frame["Fare"] / size)


def make_pipeline(experiment, model):
    if model == "LR":
        estimator = LogisticRegression(max_iter=2000, random_state=SEED)
    else:
        estimator = RandomForestClassifier(
            n_estimators=48, max_depth=9, min_samples_leaf=3, max_features="sqrt", random_state=SEED,
        )
    if experiment == "Base":
        columns = ColumnTransformer([("base", "passthrough", BASE)], verbose_feature_names_out=False)
        return Pipeline([("columns", columns), ("model", estimator)])

    impute, encoding, scaling, derived, select = CONFIGS[experiment]
    if encoding == "One-Hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    numeric = Pipeline([("fill", SimpleImputer(strategy=impute)), ("scale", SCALERS[scaling]())])
    categorical = Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", encoder)])
    columns = ColumnTransformer(
        [("numeric", numeric, NUM + (DERIVED if derived else [])), ("categorical", categorical, CAT)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("derive", FunctionTransformer(add_features) if derived else "passthrough"),
        ("columns", columns),
        ("select", SelectKBest(f_classif, k=6) if select else "passthrough"),
        ("model", estimator),
    ])


def evaluate(estimator, x, y, folds):
    scores = cross_validate(
        estimator, x, y, cv=folds, scoring=SCORING, return_train_score=True, error_score="raise",
    )
    return {
        **{name: float(scores[f"test_{name}"].mean()) for name in SCORING},
        "train_f1": float(scores["train_f1"].mean()),
        "roc_auc_sd": float(scores["test_roc_auc"].std(ddof=1)),
        "f1_sd": float(scores["test_f1"].std(ddof=1)),
    }


def holdout_scores(estimator, x, y):
    predicted = estimator.predict(x)
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, estimator.predict_proba(x)[:, 1]),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def auto_search(x, y, folds):
    candidates = []
    for experiment, model in product(CORE[1:], MODELS):
        pipeline = make_pipeline(experiment, model)
        if model == "LR":
            candidates.append({"candidate": [pipeline], "candidate__model__C": [1., 5.]})
        else:
            for depth, leaf in ((9, 3), (5, 6)):
                candidates.append({
                    "candidate": [pipeline], "candidate__model__max_depth": [depth],
                    "candidate__model__min_samples_leaf": [leaf],
                })
    search = GridSearchCV(
        Pipeline([("candidate", make_pipeline("Exp-1", "LR"))]), candidates,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring="roc_auc", error_score="raise",
    )
    return {"scores": evaluate(search, x, y, folds)}


def linear_shap(x, y):
    pipeline = make_pipeline("Exp-1", "LR").fit(x, y)
    matrix = pipeline[:-1].transform(x)
    names = pipeline.named_steps["columns"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    baseline = matrix.mean(axis=0)
    values = (matrix - baseline) * model.coef_[0]
    expected = model.intercept_[0] + baseline @ model.coef_[0]
    np.testing.assert_allclose(expected + values.sum(axis=1), model.decision_function(matrix), atol=1e-10)
    return dict(zip(names, abs(values).mean(axis=0))), float(expected)


def selected_features(x, y):
    result = {}
    for experiment in ("Exp-2", "Exp-3"):
        pipeline = make_pipeline(experiment, "LR")[:-1].fit(x, y)
        names = pipeline.named_steps["columns"].get_feature_names_out()
        select = pipeline.named_steps["select"]
        kept = sorted(select.get_support(indices=True), key=lambda i: select.scores_[i], reverse=True)
        result[experiment] = {names[i]: float(select.scores_[i]) for i in kept}
    return result


def outlier_counts(eda):
    result = {}
    for column in ("Age", "Fare"):
        observed = eda[column].dropna()
        q1, q3 = observed.quantile([.25, .75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        result[column] = {
            "count": int(((observed < low) | (observed > high)).sum()), "observed": len(observed),
            "low": float(low), "high": float(high),
        }
    return result



def save_eda_plots(eda, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for i, column in enumerate(("Age", "Fare")):
        values = eda[column].dropna()
        axes[0, i].hist(values, bins=20, color="#167b72", edgecolor="white")
        axes[0, i].set(xlabel=column, ylabel="Passengers", title=f"{column} histogram")
        axes[1, i].boxplot(values, orientation="horizontal")
        axes[1, i].set(xlabel=column, title=f"{column} boxplot")
    fig.savefig(out / "eda_distributions.png", dpi=150)
    plt.close(fig)

    columns = ["Pclass", "SibSp", "Parch", "Fare", "Age", "Survived"]
    corr = eda[columns].corr()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ax = axes[0]
    heat = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > .65 else "black")
    ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(columns)),
           yticklabels=columns, title="Pearson correlation")
    ax.tick_params(axis="x", rotation=30)
    fig.colorbar(heat, ax=ax)
    counts = eda["Survived"].value_counts().reindex([0, 1])
    bars = axes[1].bar(["0 (Died)", "1 (Survived)"], counts, color=["#496e9b", "#167b72"])
    axes[1].bar_label(bars, padding=4)
    axes[1].set(ylabel="Passengers", title="Target count", ylim=(0, counts.max() * 1.15))
    fig.savefig(out / "eda_relationships.png", dpi=150)
    plt.close(fig)


def save_importance_plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = pd.DataFrame.from_dict(summary["importance"], orient="index", columns=["mean", "sd"])
    importance = importance.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.barh(importance.index, importance["mean"], xerr=importance["sd"], color="#167b72")
    ax.set(title="Permutation importance", xlabel="Decrease in ROC-AUC (mean and SD)")
    fig.savefig(out / "permutation_importance.png", dpi=150)
    plt.close(fig)

    if "shap" in summary:
        top = pd.Series(summary["shap"]).nlargest(8).sort_values()
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        ax.barh(top.index, top, color="#496e9b")
        ax.set(title="Linear SHAP", xlabel="Mean absolute contribution to log-odds")
        fig.savefig(out / "linear_shap.png", dpi=150)
        plt.close(fig)


def main(data_path, out, bonus=True, stage="all"):
    data = pd.read_csv(data_path)
    if missing := REQUIRED - set(data):
        raise ValueError(f"Missing Titanic columns: {', '.join(sorted(missing))}")
    if len(data) < 500 or set(data["Survived"].unique()) != {0, 1}:
        raise ValueError("Expected 500 or more labeled passengers and a binary Survived column.")
    if not data["PassengerId"].is_unique or data[BASE].isna().any().any():
        raise ValueError("PassengerId must be unique and the raw Base columns must be complete.")
    out.mkdir(parents=True, exist_ok=True)

    x = data[NUM + CAT]
    y = data["Survived"]
    train, test, train_y, test_y = train_test_split(
        x, y, test_size=.2, stratify=y, random_state=SEED,
    )
    eda = data.loc[train.index]
    outliers = outlier_counts(eda)
    if stage != "train":
        save_eda_plots(eda, out)
        eda_summary = {
            "dev_rows": train.index.tolist(), "test_size": len(test), "outliers": outliers,
            "missing": eda.isna().sum().to_dict(), "target": eda["Survived"].value_counts().to_dict(),
            "duplicates": int(eda.duplicated().sum()),
        }
        (out / "eda_summary.json").write_text(
            json.dumps(eda_summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if stage == "eda":
            return
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison = pd.DataFrame([
        {"experiment": experiment, "model": model,
         **evaluate(make_pipeline(experiment, model), train, train_y, folds)}
        for experiment, model in product(("Base", *CONFIGS), MODELS)
    ])
    core = comparison[comparison["experiment"].isin(CORE)]
    winner = core.loc[core["roc_auc"].idxmax()]
    experiment, model = winner["experiment"], winner["model"]
    if model == "LR":
        grid = {"model__C": [.1, .3, 1., 3., 10.]}
    else:
        grid = {"model__max_depth": [5, 9, 13], "model__min_samples_leaf": [2, 3, 6]}
    search = GridSearchCV(
        make_pipeline(experiment, model), grid, scoring="roc_auc", cv=folds, error_score="raise",
    ).fit(train, train_y)
    final = search.best_estimator_
    holdout = holdout_scores(final, test, test_y)
    importance = permutation_importance(final, test, test_y, scoring="roc_auc", n_repeats=25, random_state=SEED)

    summary = {
        "shape": data.shape, "split_sizes": [len(train), len(test)],
        "target": data["Survived"].value_counts().to_dict(),
        "missing": data.isna().sum().to_dict(), "duplicates": int(data.duplicated().sum()),
        "eda_target": eda["Survived"].value_counts().to_dict(),
        "sex_survival": eda.groupby("Sex")["Survived"].mean().to_dict(),
        "class_survival": eda.groupby("Pclass")["Survived"].mean().to_dict(),
        "ticket_overlap": len(set(eda["Ticket"]) & set(data.loc[test.index, "Ticket"])),
        "outliers": outliers, "selections": selected_features(train, train_y),
        "winner": winner.to_dict(), "best_params": search.best_params_, "tuned_cv_auc": float(search.best_score_),
        "tuning": list(zip(search.cv_results_["params"], map(float, search.cv_results_["mean_test_score"]))),
        "holdout": holdout,
        "importance": {
            name: [float(mean), float(sd)]
            for name, mean, sd in zip(test.columns, importance.importances_mean, importance.importances_std)
        },
    }
    if bonus:
        summary["shap"], summary["shap_baseline"] = linear_shap(train, train_y)
        summary["automl"] = auto_search(train, train_y, folds)

    save_importance_plots(summary, out)
    comparison.to_csv(out / "cv_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"row": train.index}).to_csv(out / "eda_rows.csv", index=False)
    print(f"Data shape: {data.shape}")
    print(core.to_string(index=False))
    print("Holdout:", holdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--skip-bonus", action="store_true")
    parser.add_argument("--stage", choices=("all", "eda", "train"), default="all",
                        help="Run everything, run EDA only, or run training after EDA.")
    args = parser.parse_args()
    main(args.train_csv, args.out, bonus=not args.skip_bonus, stage=args.stage)

