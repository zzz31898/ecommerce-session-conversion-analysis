"""Create deployment-safe diagnostics for the trained champion models.

This script uses the original train/test split and champion training artifacts
to create out-of-sample scores for every historical session. It also exports
calibration diagnostics, test-set lift tables, operating points, bootstrap
confidence intervals, and a feature-availability register.

Run this with the same Python environment used for 06_购买倾向建模.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing modeling dependency. Run: "
        "python -m pip install -r requirements-model.txt"
    ) from exc


RANDOM_STATE = 42
TARGET = "RevenueFlag"
SCENARIO_PREFIX = {
    "early_propensity": "BehaviorExclPageValues",
    "late_session": "PageValuesBenchmark",
}
SCENARIO_NAME = {
    "early_propensity": "排除PageValues的会话行为模型",
    "late_session": "PageValues回顾性基准模型",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "online_shoppers_analysis_ready_v2.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_dir / "outputs" / "models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "models" / "enhanced_evaluation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "06_模型评估补充报告.md",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of stratified bootstrap resamples for champion metrics.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    if "SessionID" not in data or TARGET not in data:
        raise ValueError("Input data must contain SessionID and RevenueFlag.")
    if data["SessionID"].duplicated().any():
        raise ValueError("SessionID must be unique.")
    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()) <= {0, 1}:
        raise ValueError("RevenueFlag must contain only 0 and 1.")
    return data


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def pct(value: float) -> str:
    return f"{value:.2%}"


def number(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def score_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "ROCAUC": roc_auc_score(y_true, scores),
        "PRAUC": average_precision_score(y_true, scores),
        "BrierScore": brier_score_loss(y_true, scores),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(y_true == 1)
    negative_index = np.flatnonzero(y_true == 0)
    rows = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [
                rng.choice(positive_index, size=len(positive_index), replace=True),
                rng.choice(negative_index, size=len(negative_index), replace=True),
            ]
        )
        rows.append(score_metrics(y_true[sampled], scores[sampled], threshold))
    distribution = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "Metric": distribution.columns,
            "Lower95": distribution.quantile(0.025).to_numpy(),
            "Upper95": distribution.quantile(0.975).to_numpy(),
            "BootstrapIterations": iterations,
        }
    )


def calibration_bins(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    ranking = pd.Series(scores).rank(method="first")
    group_count = min(bins, len(scores))
    score_bin = pd.qcut(ranking, q=group_count, labels=False) + 1
    table = pd.DataFrame({"ScoreBinLowToHigh": score_bin, "Actual": y_true, "Score": scores})
    result = (
        table.groupby("ScoreBinLowToHigh", observed=True)
        .agg(
            Sessions=("Actual", "count"),
            Purchases=("Actual", "sum"),
            MeanScore=("Score", "mean"),
            ObservedConversionRate=("Actual", "mean"),
        )
        .reset_index()
    )
    result["CalibrationGapObservedMinusScore"] = (
        result["ObservedConversionRate"] - result["MeanScore"]
    )
    return result


def test_lift_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"Actual": y_true, "Score": scores}).sort_values(
        "Score", ascending=False, kind="mergesort"
    )
    frame["Rank"] = np.arange(1, len(frame) + 1)
    frame["ScoreDecileHighToLow"] = np.ceil(frame["Rank"] / len(frame) * 10).astype(int)
    baseline = frame["Actual"].mean()
    result = (
        frame.groupby("ScoreDecileHighToLow", observed=True)
        .agg(
            Sessions=("Actual", "count"),
            Purchases=("Actual", "sum"),
            MeanScore=("Score", "mean"),
            ObservedConversionRate=("Actual", "mean"),
        )
        .reset_index()
        .sort_values("ScoreDecileHighToLow")
    )
    result["LiftVsTestBaseline"] = result["ObservedConversionRate"] / baseline
    result["CumulativeSessions"] = result["Sessions"].cumsum()
    result["CumulativePurchases"] = result["Purchases"].cumsum()
    result["CumulativeRecall"] = result["CumulativePurchases"] / frame["Actual"].sum()
    return result


def operating_points(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"Actual": y_true, "Score": scores}).sort_values(
        "Score", ascending=False, kind="mergesort"
    )
    baseline = frame["Actual"].mean()
    rows = []
    for coverage in [0.05, 0.10, 0.20, 0.30]:
        selected_count = max(1, math.ceil(len(frame) * coverage))
        selected = frame.head(selected_count)
        purchases = int(selected["Actual"].sum())
        precision = purchases / selected_count
        rows.append(
            {
                "SelectedTopPercent": coverage,
                "SelectedSessions": selected_count,
                "ScoreThresholdAtCoverage": selected["Score"].iloc[-1],
                "PurchasesCaptured": purchases,
                "Precision": precision,
                "Recall": purchases / int(frame["Actual"].sum()),
                "LiftVsTestBaseline": precision / baseline,
            }
        )
    return pd.DataFrame(rows)


def feature_availability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "FeatureGroup": "会话入口或稳定属性",
                "Fields": "Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend",
                "Availability": "需由埋点/会话创建时确认",
                "AllowedForRealTimeEarlyIntervention": "待确认",
                "Decision": "当前仅可作为会话级分析特征，不能假定实时可用。",
            },
            {
                "FeatureGroup": "整段会话聚合行为",
                "Fields": "Administrative, Informational, ProductRelated, *_Duration, BounceRates, ExitRates, SpecialDay",
                "Availability": "会话进行中或结束后才逐步/完整形成",
                "AllowedForRealTimeEarlyIntervention": "否",
                "Decision": "当前数据没有观察窗口和事件时间戳，因此不能称为早期实时特征。",
            },
            {
                "FeatureGroup": "PageValues",
                "Fields": "PageValues",
                "Availability": "与交易归因关系密切，实际生成时点未验证",
                "AllowedForRealTimeEarlyIntervention": "否",
                "Decision": "仅作为回顾性基准；在获得字段生成时点证据前不进入部署模型。",
            },
        ]
    )


def build_report(
    champion_summary: pd.DataFrame,
    confidence: pd.DataFrame,
    calibration: pd.DataFrame,
    lift: pd.DataFrame,
    operating: pd.DataFrame,
    availability: pd.DataFrame,
    high_potential_count: int,
) -> str:
    behavior = champion_summary.loc[
        champion_summary["Scenario"].eq("early_propensity")
    ].iloc[0]
    benchmark = champion_summary.loc[
        champion_summary["Scenario"].eq("late_session")
    ].iloc[0]
    behavior_lift = lift.loc[
        (lift["Scenario"].eq("early_propensity"))
        & (lift["ScoreDecileHighToLow"].eq(1))
    ].iloc[0]
    behavior_operating = operating.loc[operating["Scenario"].eq("early_propensity")]

    lines = [
        "# 购买倾向模型评估补充报告",
        "",
        "## 1. 本报告修正的模型定位",
        "",
        "- 排除 `PageValues` 的模型使用的仍是整段会话聚合行为，因此它是“排除PageValues的会话行为模型”，不是实时早期干预模型。",
        "- 包含 `PageValues` 的模型是回顾性性能基准。该字段与交易归因关系密切，在真实生成时点未被验证前，不用于部署评分。",
        "- 历史会话评分统一使用训练集折外分数或测试集留出分数，不使用模型对自身训练记录产生的分数。",
        "",
        "## 2. 优胜模型测试集诊断",
        "",
    ]
    summary_rows = [
        [
            row["ScenarioName"],
            row["ModelName"],
            number(row["PRAUC"]),
            number(row["ROCAUC"]),
            number(row["BrierScore"]),
            pct(row["Precision"]),
            pct(row["Recall"]),
            pct(row["F1"]),
        ]
        for _, row in champion_summary.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["场景", "模型", "PR-AUC", "ROC-AUC", "Brier Score", "Precision", "Recall", "F1"],
            summary_rows,
        )
    )
    lines.extend(
        [
            "",
            "Brier Score和校准分箱用于诊断分数与实际转化率的对应程度；当前输出字段统一称为“倾向分数”，不将其直接表述为经过校准的购买概率。",
            "",
            "## 3. 优胜模型测试集置信区间",
            "",
        ]
    )
    ci_rows = [
        [
            row["ScenarioName"],
            row["Metric"],
            number(row["Lower95"]),
            number(row["Upper95"]),
            int(row["BootstrapIterations"]),
        ]
        for _, row in confidence.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["场景", "指标", "95%下限", "95%上限", "Bootstrap次数"], ci_rows
        )
    )

    lines.extend(
        [
            "",
            "## 4. 排除PageValues模型的测试集Lift与触达档位",
            "",
            f"测试集最高分10%会话的实际转化率为 {pct(behavior_lift['ObservedConversionRate'])}，测试集整体转化率的Lift为 {number(behavior_lift['LiftVsTestBaseline'], 2)} 倍。该结果来自未参与训练的测试记录。",
            "",
        ]
    )
    operating_rows = [
        [
            pct(row["SelectedTopPercent"]),
            int(row["SelectedSessions"]),
            number(row["ScoreThresholdAtCoverage"]),
            int(row["PurchasesCaptured"]),
            pct(row["Precision"]),
            pct(row["Recall"]),
            number(row["LiftVsTestBaseline"], 2),
        ]
        for _, row in behavior_operating.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["触达Top比例", "会话数", "最低分数", "捕获购买", "Precision", "Recall", "Lift"],
            operating_rows,
        )
    )

    lines.extend(["", "## 5. 特征可用性边界", ""])
    availability_rows = [
        [
            row["FeatureGroup"],
            row["Availability"],
            row["AllowedForRealTimeEarlyIntervention"],
            row["Decision"],
        ]
        for _, row in availability.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["特征组", "可用时点", "可用于实时早期干预", "处理决定"], availability_rows
        )
    )

    lines.extend(
        [
            "",
            "## 6. 历史高潜会话评分",
            "",
            f"未购买会话中，使用排除PageValues模型的历史样本外分数选出的Top 10%共有 {high_potential_count:,} 条。评分来源字段区分 `train_oof` 与 `test_holdout`，确保每条历史会话的标签没有被用于拟合生成该条分数的模型。",
            "",
            "该群体仍是回顾性分析对象，不能用其自身的已知未购买结果验证Lift；运营Lift应始终以测试集或未来新会话验证。",
            "",
            "## 7. Power BI和SQL使用规则",
            "",
            "1. 历史分析仅连接 `out_of_sample_session_scores.csv`，使用 `BehaviorExclPageValuesOutOfSampleScore` 和 `HistoricalHighPotentialTop10Percent`。",
            "2. 不在Power BI中把分数显示为“购买概率”；使用“购买倾向分数”或“风险排序”。",
            "3. 未来新会话由 `champion__early_propensity.joblib` 评分；该模型仍要求输入字段在实际评分时已可获得。",
            "4. `PageValuesBenchmarkOutOfSampleScore` 仅作回顾性对照，不进入业务触达规则。",
            "",
            "## 8. 输出文件",
            "",
            "- `out_of_sample_session_scores.csv`",
            "- `calibration_bins.csv`",
            "- `test_decile_lift.csv`",
            "- `operating_points.csv`",
            "- `bootstrap_confidence_intervals.csv`",
            "- `feature_availability.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100.")

    data = load_data(input_path)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    champions = pd.read_csv(model_dir / "champion_models.csv", encoding="utf-8-sig")
    test_size = float(metadata["test_size"])
    cv_folds = int(metadata["cv_folds"])
    train_index, test_index = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=int(metadata["random_state"]),
        stratify=data[TARGET],
    )
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    y_train = train_data[TARGET].to_numpy()
    y_test = test_data[TARGET].to_numpy()
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=int(metadata["random_state"]),
    )

    score_output = data[["SessionID", TARGET]].copy()
    champion_summary_rows = []
    calibration_tables = []
    lift_tables = []
    operating_tables = []
    confidence_tables = []

    for scenario_key in ["early_propensity", "late_session"]:
        champion = champions.loc[champions["Scenario"].eq(scenario_key)].iloc[0]
        model_key = champion["Model"]
        artifact_path = model_dir / "artifacts" / f"{scenario_key}__{model_key}.joblib"
        artifact = joblib.load(artifact_path)
        pipeline = artifact["pipeline"]
        threshold = float(artifact["threshold"])
        features = artifact["features"]["all"]

        X_train = train_data[features]
        X_test = test_data[features]
        print(f"Generating OOF scores for {scenario_key} / {model_key} ...", flush=True)
        oof_scores = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=args.n_jobs,
        )[:, 1]
        test_scores = pipeline.predict_proba(X_test)[:, 1]

        existing_test = pd.read_csv(model_dir / "test_predictions.csv", encoding="utf-8-sig")
        existing_test = existing_test.loc[
            (existing_test["Scenario"].eq(scenario_key))
            & (existing_test["Model"].eq(model_key))
        ].sort_values("SessionID")
        expected_session_ids = np.sort(test_data["SessionID"].to_numpy())
        if not np.array_equal(existing_test["SessionID"].to_numpy(), expected_session_ids):
            raise AssertionError("Stored test prediction SessionIDs do not match recreated split.")

        test_order = np.argsort(test_data["SessionID"].to_numpy())
        if not np.allclose(
            existing_test["PurchaseProbability"].to_numpy(),
            test_scores[test_order],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise AssertionError("Stored test predictions do not match champion artifact scores.")

        historical_scores = pd.Series(index=data.index, dtype=float)
        historical_source = pd.Series(index=data.index, dtype=object)
        historical_scores.loc[train_data.index] = oof_scores
        historical_source.loc[train_data.index] = "train_oof"
        historical_scores.loc[test_data.index] = test_scores
        historical_source.loc[test_data.index] = "test_holdout"
        if historical_scores.isna().any() or historical_source.isna().any():
            raise AssertionError("Historical out-of-sample scores are incomplete.")

        prefix = SCENARIO_PREFIX[scenario_key]
        score_output[f"{prefix}OutOfSampleScore"] = historical_scores.to_numpy()
        score_output[f"{prefix}ScoreSource"] = historical_source.to_numpy()

        metrics = score_metrics(y_test, test_scores, threshold)
        champion_summary_rows.append(
            {
                "Scenario": scenario_key,
                "ScenarioName": SCENARIO_NAME[scenario_key],
                "Model": model_key,
                "ModelName": champion["ModelName"],
                "Threshold": threshold,
                **metrics,
            }
        )
        calibration = calibration_bins(y_test, test_scores)
        calibration.insert(0, "Model", model_key)
        calibration.insert(0, "Scenario", scenario_key)
        calibration_tables.append(calibration)

        lift = test_lift_table(y_test, test_scores)
        lift.insert(0, "Model", model_key)
        lift.insert(0, "Scenario", scenario_key)
        lift_tables.append(lift)

        operating = operating_points(y_test, test_scores)
        operating.insert(0, "Model", model_key)
        operating.insert(0, "Scenario", scenario_key)
        operating_tables.append(operating)

        confidence = bootstrap_intervals(
            y_test,
            test_scores,
            threshold,
            args.bootstrap_iterations,
            seed=RANDOM_STATE + (0 if scenario_key == "early_propensity" else 1000),
        )
        confidence.insert(0, "ScenarioName", SCENARIO_NAME[scenario_key])
        confidence.insert(0, "Scenario", scenario_key)
        confidence_tables.append(confidence)

    behavior_score = score_output["BehaviorExclPageValuesOutOfSampleScore"]
    unpurchased = score_output[TARGET].eq(0)
    score_output["BehaviorExclPageValuesPercentileAmongUnpurchased"] = np.nan
    score_output.loc[
        unpurchased, "BehaviorExclPageValuesPercentileAmongUnpurchased"
    ] = behavior_score.loc[unpurchased].rank(method="average", pct=True)
    score_output["HistoricalHighPotentialTop10Percent"] = (
        unpurchased
        & score_output["BehaviorExclPageValuesPercentileAmongUnpurchased"].ge(0.90)
    )

    champion_summary = pd.DataFrame(champion_summary_rows)
    calibration = pd.concat(calibration_tables, ignore_index=True)
    lift = pd.concat(lift_tables, ignore_index=True)
    operating = pd.concat(operating_tables, ignore_index=True)
    confidence = pd.concat(confidence_tables, ignore_index=True)
    availability = feature_availability()

    outputs = {
        "out_of_sample_session_scores.csv": score_output,
        "calibration_bins.csv": calibration,
        "test_decile_lift.csv": lift,
        "operating_points.csv": operating,
        "bootstrap_confidence_intervals.csv": confidence,
        "feature_availability.csv": availability,
        "champion_diagnostic_metrics.csv": champion_summary,
    }
    for filename, table in outputs.items():
        write_csv(table, output_dir / filename)

    report = build_report(
        champion_summary,
        confidence,
        calibration,
        lift,
        operating,
        availability,
        int(score_output["HistoricalHighPotentialTop10Percent"].sum()),
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Historical out-of-sample scores: {len(score_output):,}")
    print(f"High-potential historical unpurchased sessions: {int(score_output['HistoricalHighPotentialTop10Percent'].sum()):,}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
