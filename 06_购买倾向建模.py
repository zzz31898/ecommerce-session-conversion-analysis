"""Train and evaluate session purchase-propensity models.

Two deployment scenarios are evaluated:
1. early_propensity: excludes PageValues for earlier intervention.
2. late_session: includes PageValues for end-of-session prediction.

Each scenario compares logistic regression, a decision tree, and a random
forest. Hyperparameters and decision thresholds are selected using training
data only. The untouched test set is used once for final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
        cross_val_predict,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing modeling dependency. Run: "
        "python -m pip install -r requirements-model.txt"
    ) from exc


RANDOM_STATE = 42
TARGET = "RevenueFlag"

SKEWED_NUMERIC_BASE = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]

RATE_NUMERIC = ["BounceRates", "ExitRates", "SpecialDay"]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

SCENARIOS = {
    "early_propensity": {
        "name": "排除PageValues的会话行为模型",
        "description": "排除PageValues的会话级行为模型；整段会话聚合特征不保证实时可用。",
        "include_page_values": False,
    },
    "late_session": {
        "name": "PageValues回顾性基准模型",
        "description": "包含PageValues的回顾性性能基准；字段生成时点未确认前不用于部署。",
        "include_page_values": True,
    },
}

MODEL_NAMES = {
    "logistic_regression": "逻辑回归",
    "decision_tree": "决策树",
    "random_forest": "随机森林",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "online_shoppers_analysis_ready_v2.csv",
        help="Path to the analysis-ready CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "models",
        help="Directory for model outputs and artifacts.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "06_购买倾向模型报告.md",
        help="Path for the generated Markdown report.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for the untouched test set.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by grid search and out-of-fold prediction.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids and three folds for a faster smoke run.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "SessionID",
        TARGET,
        "Revenue",
        *SKEWED_NUMERIC_BASE,
        *RATE_NUMERIC,
        *CATEGORICAL_FEATURES,
        "PageValues",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("The input dataset contains no rows.")
    if data["SessionID"].duplicated().any():
        raise ValueError("SessionID must be unique.")
    if data[list(required)].isna().any().any():
        raise ValueError("Required modeling fields contain missing values.")

    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()) <= {0, 1}:
        raise ValueError("RevenueFlag must contain only 0 and 1.")
    if data[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required.")

    for column in [*SKEWED_NUMERIC_BASE, *RATE_NUMERIC, "PageValues"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data[[*SKEWED_NUMERIC_BASE, *RATE_NUMERIC, "PageValues"]] < 0).any().any():
        raise ValueError("Modeling numeric features must be nonnegative.")
    return data


def scenario_features(include_page_values: bool) -> dict[str, list[str]]:
    skewed = [*SKEWED_NUMERIC_BASE]
    if include_page_values:
        skewed.append("PageValues")
    return {
        "skewed_numeric": skewed,
        "rate_numeric": [*RATE_NUMERIC],
        "categorical": [*CATEGORICAL_FEATURES],
        "all": [*skewed, *RATE_NUMERIC, *CATEGORICAL_FEATURES],
    }


def build_preprocessor(features: dict[str, list[str]]) -> ColumnTransformer:
    skewed_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    rate_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("skewed", skewed_pipeline, features["skewed_numeric"]),
            ("rates", rate_pipeline, features["rate_numeric"]),
            ("categorical", categorical_pipeline, features["categorical"]),
        ],
        remainder="drop",
        sparse_threshold=0.30,
        verbose_feature_names_out=True,
    )


def model_specs(quick: bool) -> dict[str, dict[str, object]]:
    random_forest_trees = 100 if quick else 300
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "classifier__C": [0.1, 1.0] if quick else [0.03, 0.1, 0.3, 1.0, 3.0],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "classifier__max_depth": [4, 8] if quick else [3, 5, 8, 12],
                "classifier__min_samples_leaf": [30] if quick else [15, 30, 60],
                "classifier__min_samples_split": [20],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=random_forest_trees,
                class_weight="balanced_subsample",
                max_features="sqrt",
                bootstrap=True,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "classifier__max_depth": [8, None] if quick else [6, 10, 16, None],
                "classifier__min_samples_leaf": [10] if quick else [5, 15, 30],
            },
        },
    }


def select_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    maximum = float(np.nanmax(f1_values))
    candidate_indices = np.flatnonzero(np.isclose(f1_values, maximum, rtol=1e-10, atol=1e-12))
    selected_index = int(candidate_indices[-1])
    return float(thresholds[selected_index]), maximum


def evaluate_predictions(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROCAUC": roc_auc_score(y_true, probabilities),
        "PRAUC": average_precision_score(y_true, probabilities),
        "TrueNegative": int(tn),
        "FalsePositive": int(fp),
        "FalseNegative": int(fn),
        "TruePositive": int(tp),
    }


def clean_feature_name(name: str) -> str:
    for prefix in ["skewed__", "rates__", "categorical__"]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def feature_importance_table(
    estimator: Pipeline,
    scenario_key: str,
    model_key: str,
) -> pd.DataFrame:
    preprocessor = estimator.named_steps["preprocessor"]
    classifier = estimator.named_steps["classifier"]
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]

    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        importance_type = "standardized_coefficient"
        directions = np.where(values >= 0, "positive", "negative")
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        importance_type = "impurity_importance"
        directions = np.full(len(values), "non_directional")
    else:
        return pd.DataFrame()

    if len(feature_names) != len(values):
        raise AssertionError("Feature-name and importance lengths do not match.")
    table = pd.DataFrame(
        {
            "Scenario": scenario_key,
            "ScenarioName": SCENARIOS[scenario_key]["name"],
            "Model": model_key,
            "ModelName": MODEL_NAMES[model_key],
            "Feature": feature_names,
            "ImportanceValue": values,
            "AbsoluteImportance": np.abs(values),
            "Direction": directions,
            "ImportanceType": importance_type,
        }
    )
    return table.sort_values("AbsoluteImportance", ascending=False).reset_index(drop=True)


def compact_cv_results(
    search: GridSearchCV,
    scenario_key: str,
    model_key: str,
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    columns = [
        "rank_test_pr_auc",
        "mean_test_pr_auc",
        "std_test_pr_auc",
        "mean_test_roc_auc",
        "mean_test_f1",
        "mean_fit_time",
        "params",
    ]
    compact = results[columns].copy()
    compact.insert(0, "Model", model_key)
    compact.insert(0, "Scenario", scenario_key)
    compact["params"] = compact["params"].map(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True))
    return compact.sort_values("rank_test_pr_auc")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: dict[str, list[str]],
    scenario_key: str,
    model_key: str,
    specification: dict[str, object],
    cv: StratifiedKFold,
    n_jobs: int,
) -> dict[str, object]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", clone(specification["estimator"])),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=specification["grid"],
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"},
        refit="pr_auc",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )

    started = perf_counter()
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_

    oof_probabilities = cross_val_predict(
        clone(best_estimator),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]
    threshold, oof_f1 = select_f1_threshold(y_train, oof_probabilities)
    test_probabilities = best_estimator.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, test_probabilities, threshold)
    elapsed = perf_counter() - started

    return {
        "scenario": scenario_key,
        "model": model_key,
        "estimator": best_estimator,
        "features": deepcopy(features),
        "threshold": threshold,
        "oof_f1": oof_f1,
        "cv_pr_auc": float(search.best_score_),
        "best_params": deepcopy(search.best_params_),
        "metrics": metrics,
        "test_probabilities": test_probabilities,
        "elapsed_seconds": elapsed,
        "cv_results": compact_cv_results(search, scenario_key, model_key),
        "feature_importance": feature_importance_table(best_estimator, scenario_key, model_key),
    }


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_number(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def build_report(
    input_path: Path,
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    champions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    propensity: pd.DataFrame,
    train_size: int,
    test_size: int,
    cv_folds: int,
    quick: bool,
) -> str:
    baseline = data[TARGET].mean()
    early_champion = champions.loc[champions["Scenario"].eq("early_propensity")].iloc[0]
    late_champion = champions.loc[champions["Scenario"].eq("late_session")].iloc[0]
    high_potential_count = int(propensity["HighPotentialTop10Percent"].sum())

    lines = [
        "# 电商网站会话购买倾向模型报告",
        "",
        "## 1. 建模设计",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 输入文件：`{input_path.name}`",
        f"- 输入文件 SHA-256：`{sha256(input_path)}`",
        f"- 总样本：{len(data):,} 个会话，购买会话占比 {format_pct(baseline)}",
        f"- 训练集：{train_size:,} 个会话；测试集：{test_size:,} 个会话",
        f"- 交叉验证：{cv_folds}折分层交叉验证",
        f"- 运行模式：{'快速验证' if quick else '完整训练'}",
        "- 调参目标：训练集交叉验证 PR-AUC。",
        "- 阈值选择：使用最佳模型的训练集折外预测，以最大化F1；测试集不参与调参或阈值选择。",
        "- 类别不平衡：三个模型均使用类别权重；评价不以准确率作为唯一指标。",
        "- 排除PageValues模型：仍使用整段会话聚合行为，不应称为实时早期模型。",
        "- PageValues模型：仅作回顾性性能基准；字段生成时点确认前不用于部署评分。",
        "- 未使用 `SessionID`、`Revenue`、派生行为分组或高跳出/高退出标记作为模型特征。",
        "",
        "## 2. 测试集表现",
        "",
        f"随机分类器的PR-AUC基线约等于购买率，即 {format_pct(baseline)}。",
        "",
    ]
    metric_rows = []
    for _, row in metrics.sort_values(["Scenario", "TestPRAUC"], ascending=[True, False]).iterrows():
        metric_rows.append(
            [
                row["ScenarioName"],
                row["ModelName"],
                format_number(row["CVPRAUC"]),
                format_number(row["Threshold"]),
                format_pct(row["TestPrecision"]),
                format_pct(row["TestRecall"]),
                format_pct(row["TestF1"]),
                format_number(row["TestROCAUC"]),
                format_number(row["TestPRAUC"]),
                format_pct(row["TestAccuracy"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["场景", "模型", "CV PR-AUC", "阈值", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Accuracy"],
            metric_rows,
        )
    )

    lines.extend(["", "## 3. 场景优胜模型", ""])
    champion_rows = []
    for _, row in champions.iterrows():
        champion_rows.append(
            [
                row["ScenarioName"],
                row["ModelName"],
                format_number(row["CVPRAUC"]),
                format_number(row["TestPRAUC"]),
                format_number(row["TestROCAUC"]),
                format_pct(row["TestF1"]),
                format_number(row["Threshold"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["场景", "优胜模型", "CV PR-AUC", "测试PR-AUC", "测试ROC-AUC", "测试F1", "决策阈值"],
            champion_rows,
        )
    )
    lines.extend(
        [
            "",
            f"早期场景根据训练集CV PR-AUC选择了{early_champion['ModelName']}；会话末期场景选择了{late_champion['ModelName']}。优胜模型依据训练集交叉验证选择，而不是根据测试集结果倒推。",
            "",
            f"加入 `PageValues` 后，优胜模型测试集PR-AUC从 {format_number(early_champion['TestPRAUC'])} 变为 {format_number(late_champion['TestPRAUC'])}。该差异反映可用信息时点不同，不能简单解释为末期模型更适合早期运营。",
            "",
            "## 4. 优胜模型混淆矩阵",
            "",
        ]
    )
    confusion_rows = []
    for _, row in champions.iterrows():
        confusion_rows.append(
            [
                row["ScenarioName"],
                int(row["TrueNegative"]),
                int(row["FalsePositive"]),
                int(row["FalseNegative"]),
                int(row["TruePositive"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["场景", "TN", "FP", "FN", "TP"], confusion_rows
        )
    )

    lines.extend(["", "## 5. 优胜模型重要特征", ""])
    for scenario_key in ["early_propensity", "late_session"]:
        champion = champions.loc[champions["Scenario"].eq(scenario_key)].iloc[0]
        subset = feature_importance.loc[
            feature_importance["Scenario"].eq(scenario_key)
            & feature_importance["Model"].eq(champion["Model"])
        ].head(12)
        lines.extend([f"### 5.{'1' if scenario_key == 'early_propensity' else '2'} {champion['ScenarioName']}", ""])
        importance_rows = [
            [
                row["Feature"],
                format_number(row["ImportanceValue"], 4),
                row["Direction"],
                row["ImportanceType"],
            ]
            for _, row in subset.iterrows()
        ]
        lines.extend(
            markdown_table(
                ["特征", "重要性值", "方向", "类型"], importance_rows
            )
        )
        lines.append("")

    lines.extend(
        [
            "逻辑回归的重要性值是标准化特征上的系数，具有方向；树模型使用基于不纯度的重要性，只能表示相对贡献，不能解释为因果影响。独热编码后的类别水平会分别显示。",
            "",
            "## 6. 高潜未购买会话",
            "",
            f"使用完整数据重新拟合排除PageValues场景优胜流水线后，对当前会话生成倾向分数。未购买会话中分数最高的10%共有 {high_potential_count:,} 条，字段 `HighPotentialTop10Percent` 属于训练内历史评分。",
            "",
            "该评分不能用于证明历史高潜群体的实际Lift。请运行 `06_模型评估增强.py`，使用折外/测试集分数与测试集Lift表替代本节评分表。",
            "",
            "## 7. 使用建议",
            "",
            "1. Power BI历史分析应使用增强脚本输出的样本外评分，不使用本脚本生成的训练内 `EarlyPropensityScore`。",
            "2. `LatePropensityScore` 仅作PageValues回顾性基准对照，不作为业务触达或部署评分。",
            "3. 实际触达阈值应结合运营成本、误触达成本和可承接量重新设定，当前阈值以训练集F1最大化为目标。",
            "4. SQL阶段把 `session_purchase_propensity.csv` 载入独立预测表，通过 `SessionID` 与会话事实表关联。",
            "5. 获得新时间段数据后，应按时间外样本重新验证模型，不能只依赖当前随机测试集。",
            "",
            "## 8. 限制",
            "",
            "- 当前数据没有真实用户ID，模型预测的是会话购买倾向，不是用户生命周期价值。",
            "- 随机分层切分不能完全模拟未来月份部署；缺少年份和时间戳，暂时无法进行严格时间外验证。",
            "- 类别编码没有业务名称，特征解释只能使用 `TrafficType`、`Browser` 等编号。",
            "- 树模型的不纯度重要性可能偏向可分裂点更多的变量，应结合逻辑回归方向和后续业务验证阅读。",
            "",
            "## 9. 输出文件",
            "",
            "结果位于 `outputs/models/`：",
            "",
            "- `model_metrics.csv`",
            "- `champion_models.csv`",
            "- `confusion_matrices.csv`",
            "- `cross_validation_results.csv`",
            "- `test_predictions.csv`",
            "- `feature_importance.csv`",
            "- `session_purchase_propensity.csv`",
            "- `model_metadata.json`",
            "- `artifacts/*.joblib`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.test_size < 0.5:
        raise ValueError("--test-size must be greater than 0 and less than 0.5.")
    cv_folds = 3 if args.quick else args.cv_folds
    if cv_folds < 3:
        raise ValueError("--cv-folds must be at least 3.")

    data = load_data(input_path)
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()
    y_train = train_data[TARGET]
    y_test = test_data[TARGET]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    trained: dict[tuple[str, str], dict[str, object]] = {}
    metrics_rows = []
    prediction_tables = []
    confusion_rows = []
    cv_tables = []
    importance_tables = []
    specifications = model_specs(args.quick)

    for scenario_key, scenario in SCENARIOS.items():
        features = scenario_features(bool(scenario["include_page_values"]))
        X_train = train_data[features["all"]]
        X_test = test_data[features["all"]]

        for model_key, specification in specifications.items():
            print(f"Training {scenario_key} / {model_key} ...", flush=True)
            result = train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                features,
                scenario_key,
                model_key,
                specification,
                cv,
                args.n_jobs,
            )
            trained[(scenario_key, model_key)] = result
            evaluation = result["metrics"]
            metrics_rows.append(
                {
                    "Scenario": scenario_key,
                    "ScenarioName": scenario["name"],
                    "Model": model_key,
                    "ModelName": MODEL_NAMES[model_key],
                    "CVPRAUC": result["cv_pr_auc"],
                    "OOFBestF1": result["oof_f1"],
                    "Threshold": evaluation["Threshold"],
                    "TestAccuracy": evaluation["Accuracy"],
                    "TestPrecision": evaluation["Precision"],
                    "TestRecall": evaluation["Recall"],
                    "TestF1": evaluation["F1"],
                    "TestROCAUC": evaluation["ROCAUC"],
                    "TestPRAUC": evaluation["PRAUC"],
                    "TrueNegative": evaluation["TrueNegative"],
                    "FalsePositive": evaluation["FalsePositive"],
                    "FalseNegative": evaluation["FalseNegative"],
                    "TruePositive": evaluation["TruePositive"],
                    "BestParams": json.dumps(result["best_params"], ensure_ascii=False, sort_keys=True),
                    "ElapsedSeconds": result["elapsed_seconds"],
                }
            )

            predictions = (result["test_probabilities"] >= result["threshold"]).astype(int)
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "Scenario": scenario_key,
                        "Model": model_key,
                        "SessionID": test_data["SessionID"].to_numpy(),
                        "ActualRevenueFlag": y_test.to_numpy(),
                        "PurchaseProbability": result["test_probabilities"],
                        "DecisionThreshold": result["threshold"],
                        "PredictedRevenueFlag": predictions,
                        "Split": "test",
                    }
                )
            )
            confusion_rows.extend(
                [
                    {
                        "Scenario": scenario_key,
                        "Model": model_key,
                        "ActualClass": actual,
                        "PredictedClass": predicted,
                        "Count": int(
                            confusion_matrix(y_test, predictions, labels=[0, 1])[actual, predicted]
                        ),
                    }
                    for actual in [0, 1]
                    for predicted in [0, 1]
                ]
            )
            cv_tables.append(result["cv_results"])
            importance_tables.append(result["feature_importance"])

            joblib.dump(
                {
                    "pipeline": result["estimator"],
                    "threshold": result["threshold"],
                    "scenario": scenario_key,
                    "model": model_key,
                    "features": result["features"],
                    "source_sha256": sha256(input_path),
                },
                artifact_dir / f"{scenario_key}__{model_key}.joblib",
            )

    metrics = pd.DataFrame(metrics_rows)
    test_predictions = pd.concat(prediction_tables, ignore_index=True)
    confusion = pd.DataFrame(confusion_rows)
    cv_results = pd.concat(cv_tables, ignore_index=True)
    feature_importance = pd.concat(importance_tables, ignore_index=True)

    champion_rows = []
    champion_objects: dict[str, dict[str, object]] = {}
    propensity = data[["SessionID", TARGET]].copy()

    for scenario_key, scenario in SCENARIOS.items():
        scenario_metrics = metrics.loc[metrics["Scenario"].eq(scenario_key)]
        champion_metric = scenario_metrics.loc[scenario_metrics["CVPRAUC"].idxmax()].copy()
        model_key = champion_metric["Model"]
        result = trained[(scenario_key, model_key)]
        champion_objects[scenario_key] = result
        champion_rows.append(champion_metric.to_dict())

        full_estimator = clone(result["estimator"])
        feature_columns = result["features"]["all"]
        full_estimator.fit(data[feature_columns], data[TARGET])
        all_probabilities = full_estimator.predict_proba(data[feature_columns])[:, 1]
        prefix = "Early" if scenario_key == "early_propensity" else "Late"
        propensity[f"{prefix}PropensityScore"] = all_probabilities
        propensity[f"{prefix}DecisionThreshold"] = result["threshold"]
        propensity[f"{prefix}PredictedPurchase"] = (
            all_probabilities >= result["threshold"]
        ).astype(int)

        joblib.dump(
            {
                "pipeline": full_estimator,
                "threshold": result["threshold"],
                "scenario": scenario_key,
                "model": model_key,
                "features": result["features"],
                "fitted_on": "full_dataset_for_scoring",
                "source_sha256": sha256(input_path),
            },
            artifact_dir / f"champion__{scenario_key}.joblib",
        )

    champions = pd.DataFrame(champion_rows)
    unpurchased_mask = propensity[TARGET].eq(0)
    propensity["EarlyPropensityPercentileAmongUnpurchased"] = np.nan
    propensity.loc[
        unpurchased_mask, "EarlyPropensityPercentileAmongUnpurchased"
    ] = propensity.loc[unpurchased_mask, "EarlyPropensityScore"].rank(
        method="average", pct=True
    )
    propensity["HighPotentialTop10Percent"] = (
        unpurchased_mask
        & propensity["EarlyPropensityPercentileAmongUnpurchased"].ge(0.90)
    )

    outputs = {
        "model_metrics.csv": metrics,
        "champion_models.csv": champions,
        "confusion_matrices.csv": confusion,
        "cross_validation_results.csv": cv_results,
        "test_predictions.csv": test_predictions,
        "feature_importance.csv": feature_importance,
        "session_purchase_propensity.csv": propensity,
    }
    for filename, table in outputs.items():
        write_csv(table, output_dir / filename)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "rows": len(data),
        "target": TARGET,
        "positive_rate": float(data[TARGET].mean()),
        "random_state": RANDOM_STATE,
        "test_size": args.test_size,
        "cv_folds": cv_folds,
        "quick_mode": args.quick,
        "selection_metric": "average_precision_pr_auc",
        "threshold_rule": "maximize_f1_on_training_out_of_fold_predictions",
        "scenarios": SCENARIOS,
        "champions": {
            row["Scenario"]: {
                "model": row["Model"],
                "cv_pr_auc": float(row["CVPRAUC"]),
                "test_pr_auc": float(row["TestPRAUC"]),
                "threshold": float(row["Threshold"]),
                "best_params": json.loads(row["BestParams"]),
            }
            for _, row in champions.iterrows()
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = build_report(
        input_path,
        data,
        metrics,
        champions,
        feature_importance,
        propensity,
        len(train_data),
        len(test_data),
        cv_folds,
        args.quick,
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows modeled: {len(data):,}")
    print(f"Models evaluated: {len(metrics)}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
