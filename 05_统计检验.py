"""Validate observed conversion differences with inferential statistics.

Methods:
- Pearson chi-square tests and Cramer's V for categorical associations.
- Mann-Whitney U tests and rank-biserial correlation for behavior metrics.
- Benjamini-Hochberg correction within each family of multiple comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05

BEHAVIOR_METRICS = [
    "ProductRelated",
    "ProductRelated_Duration",
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]

METRIC_NAMES = {
    "ProductRelated": "产品页浏览量",
    "ProductRelated_Duration": "产品页停留时长",
    "Administrative": "管理类页面访问量",
    "Administrative_Duration": "管理类页面停留时长",
    "Informational": "信息类页面访问量",
    "Informational_Duration": "信息类页面停留时长",
    "BounceRates": "跳出率",
    "ExitRates": "退出率",
    "PageValues": "页面价值",
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
        default=project_dir / "outputs" / "statistics",
        help="Directory for statistical result tables.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "05_统计检验报告.md",
        help="Path for the Markdown statistical report.",
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
    required = {"SessionID", "RevenueFlag", "Revenue", "VisitorType", "Weekend"} | set(
        BEHAVIOR_METRICS
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("The input dataset contains no rows.")
    if data["SessionID"].duplicated().any():
        raise ValueError("SessionID is not unique.")
    if not set(data["RevenueFlag"].dropna().unique()) <= {0, 1}:
        raise ValueError("RevenueFlag must contain only 0 and 1.")
    if data[["RevenueFlag", "VisitorType", "Weekend", *BEHAVIOR_METRICS]].isna().any().any():
        raise ValueError("Required statistical fields contain missing values.")

    data["RevenueFlag"] = data["RevenueFlag"].astype(int)
    weekend = data["Weekend"].astype(str).str.upper()
    if not set(weekend.unique()) <= {"TRUE", "FALSE"}:
        raise ValueError(f"Unexpected Weekend values: {sorted(weekend.unique())}")
    data["WeekendLabel"] = weekend.map({"TRUE": "周末", "FALSE": "工作日"})
    return data


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ranked
    return pd.Series(adjusted, index=p_values.index)


def cramer_v(chi_square: float, sample_size: int, rows: int, columns: int) -> float:
    denominator = sample_size * min(rows - 1, columns - 1)
    return math.sqrt(chi_square / denominator) if denominator > 0 else math.nan


def effect_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.10:
        return "可忽略"
    if magnitude < 0.30:
        return "小"
    if magnitude < 0.50:
        return "中"
    return "大"


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def chi_square_test(
    data: pd.DataFrame, source_column: str, dimension_name: str
) -> tuple[dict[str, object], pd.DataFrame]:
    observed = pd.crosstab(data[source_column], data["RevenueFlag"]).reindex(columns=[0, 1], fill_value=0)
    chi_square, p_value, degrees_freedom, expected = stats.chi2_contingency(
        observed.to_numpy(), correction=False
    )
    expected_frame = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    residuals = (observed - expected_frame) / np.sqrt(expected_frame)
    sample_size = int(observed.to_numpy().sum())
    effect = cramer_v(chi_square, sample_size, *observed.shape)

    details = []
    for category in observed.index:
        non_purchases = int(observed.loc[category, 0])
        purchases = int(observed.loc[category, 1])
        total = non_purchases + purchases
        ci_lower, ci_upper = wilson_interval(purchases, total)
        details.append(
            {
                "Dimension": dimension_name,
                "SourceColumn": source_column,
                "Category": category,
                "NonPurchaseSessions": non_purchases,
                "PurchaseSessions": purchases,
                "TotalSessions": total,
                "ConversionRate": purchases / total,
                "ConversionCILower95": ci_lower,
                "ConversionCIUpper95": ci_upper,
                "ExpectedNonPurchase": expected_frame.loc[category, 0],
                "ExpectedPurchase": expected_frame.loc[category, 1],
                "PearsonResidualNonPurchase": residuals.loc[category, 0],
                "PearsonResidualPurchase": residuals.loc[category, 1],
            }
        )

    minimum_expected = float(expected_frame.to_numpy().min())
    expected_below_five = int((expected_frame.to_numpy() < 5).sum())
    summary = {
        "Dimension": dimension_name,
        "SourceColumn": source_column,
        "ChiSquare": chi_square,
        "DegreesOfFreedom": int(degrees_freedom),
        "PValue": p_value,
        "CramerV": effect,
        "EffectMagnitude": effect_label(effect),
        "MinimumExpectedCount": minimum_expected,
        "ExpectedCellsBelow5": expected_below_five,
        "AssumptionMet": expected_below_five == 0,
        "SampleSize": sample_size,
    }
    return summary, pd.DataFrame(details)


def visitor_pairwise_tests(data: pd.DataFrame) -> pd.DataFrame:
    categories = ["New_Visitor", "Returning_Visitor", "Other"]
    rows = []
    for category_a, category_b in combinations(categories, 2):
        subset = data.loc[data["VisitorType"].isin([category_a, category_b])]
        observed = pd.crosstab(subset["VisitorType"], subset["RevenueFlag"]).reindex(
            index=[category_a, category_b], columns=[0, 1], fill_value=0
        )
        chi_square, p_value, degrees_freedom, expected = stats.chi2_contingency(
            observed.to_numpy(), correction=False
        )
        sessions_a = int(observed.loc[category_a].sum())
        sessions_b = int(observed.loc[category_b].sum())
        purchases_a = int(observed.loc[category_a, 1])
        purchases_b = int(observed.loc[category_b, 1])
        conversion_a = purchases_a / sessions_a
        conversion_b = purchases_b / sessions_b
        odds_a = purchases_a / int(observed.loc[category_a, 0])
        odds_b = purchases_b / int(observed.loc[category_b, 0])
        effect = cramer_v(chi_square, len(subset), *observed.shape)
        rows.append(
            {
                "CategoryA": category_a,
                "CategoryB": category_b,
                "SessionsA": sessions_a,
                "SessionsB": sessions_b,
                "ConversionRateA": conversion_a,
                "ConversionRateB": conversion_b,
                "ConversionDifferenceAminusB": conversion_a - conversion_b,
                "OddsRatioAtoB": odds_a / odds_b,
                "ChiSquare": chi_square,
                "DegreesOfFreedom": int(degrees_freedom),
                "PValue": p_value,
                "CramerV": effect,
                "EffectMagnitude": effect_label(effect),
                "MinimumExpectedCount": float(expected.min()),
                "AssumptionMet": bool((expected >= 5).all()),
            }
        )
    result = pd.DataFrame(rows)
    result["AdjustedPValueBH"] = bh_adjust(result["PValue"])
    result["SignificantAfterBH"] = result["AdjustedPValueBH"].lt(ALPHA)
    return result


def weekend_effects(data: pd.DataFrame) -> pd.DataFrame:
    observed = pd.crosstab(data["WeekendLabel"], data["RevenueFlag"]).reindex(
        index=["工作日", "周末"], columns=[0, 1], fill_value=0
    )
    weekday_total = int(observed.loc["工作日"].sum())
    weekend_total = int(observed.loc["周末"].sum())
    weekday_purchases = int(observed.loc["工作日", 1])
    weekend_purchases = int(observed.loc["周末", 1])
    weekday_rate = weekday_purchases / weekday_total
    weekend_rate = weekend_purchases / weekend_total
    weekday_odds = weekday_purchases / int(observed.loc["工作日", 0])
    weekend_odds = weekend_purchases / int(observed.loc["周末", 0])
    return pd.DataFrame(
        [
            {
                "WeekendSessions": weekend_total,
                "WeekendPurchases": weekend_purchases,
                "WeekendConversionRate": weekend_rate,
                "WeekdaySessions": weekday_total,
                "WeekdayPurchases": weekday_purchases,
                "WeekdayConversionRate": weekday_rate,
                "ConversionDifferenceWeekendMinusWeekday": weekend_rate - weekday_rate,
                "RelativeRiskWeekendToWeekday": weekend_rate / weekday_rate,
                "OddsRatioWeekendToWeekday": weekend_odds / weekday_odds,
            }
        ]
    )


def mann_whitney_tests(data: pd.DataFrame) -> pd.DataFrame:
    purchased = data.loc[data["RevenueFlag"].eq(1)]
    not_purchased = data.loc[data["RevenueFlag"].eq(0)]
    rows = []
    for metric in BEHAVIOR_METRICS:
        purchase_values = purchased[metric].astype(float).to_numpy()
        non_purchase_values = not_purchased[metric].astype(float).to_numpy()
        result = stats.mannwhitneyu(
            purchase_values,
            non_purchase_values,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        pair_count = len(purchase_values) * len(non_purchase_values)
        rank_biserial = 2 * float(result.statistic) / pair_count - 1
        rows.append(
            {
                "Metric": metric,
                "MetricName": METRIC_NAMES[metric],
                "PurchaseN": len(purchase_values),
                "NonPurchaseN": len(non_purchase_values),
                "PurchaseMedian": np.median(purchase_values),
                "PurchaseQ1": np.quantile(purchase_values, 0.25),
                "PurchaseQ3": np.quantile(purchase_values, 0.75),
                "NonPurchaseMedian": np.median(non_purchase_values),
                "NonPurchaseQ1": np.quantile(non_purchase_values, 0.25),
                "NonPurchaseQ3": np.quantile(non_purchase_values, 0.75),
                "MedianDifferencePurchaseMinusNonPurchase": (
                    np.median(purchase_values) - np.median(non_purchase_values)
                ),
                "UStatistic": float(result.statistic),
                "PValue": float(result.pvalue),
                "RankBiserialCorrelation": rank_biserial,
                "EffectDirection": "购买会话更高" if rank_biserial > 0 else "购买会话更低",
                "EffectMagnitude": effect_label(rank_biserial),
            }
        )
    result = pd.DataFrame(rows)
    result["AdjustedPValueBH"] = bh_adjust(result["PValue"])
    result["SignificantAfterBH"] = result["AdjustedPValueBH"].lt(ALPHA)
    return result.sort_values(
        "RankBiserialCorrelation", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def format_p(value: float) -> str:
    if value == 0 or value < 1e-300:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


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


def build_report(
    input_path: Path,
    data: pd.DataFrame,
    chi_square: pd.DataFrame,
    contingency: pd.DataFrame,
    pairwise: pd.DataFrame,
    weekend: pd.DataFrame,
    mann_whitney: pd.DataFrame,
) -> str:
    visitor_test = chi_square.loc[chi_square["SourceColumn"].eq("VisitorType")].iloc[0]
    weekend_test = chi_square.loc[chi_square["SourceColumn"].eq("WeekendLabel")].iloc[0]
    visitor_detail = contingency.loc[contingency["SourceColumn"].eq("VisitorType")]
    weekend_detail = contingency.loc[contingency["SourceColumn"].eq("WeekendLabel")]
    weekend_effect = weekend.iloc[0]
    early_behavior = mann_whitney.loc[mann_whitney["Metric"].ne("PageValues")]
    page_values = mann_whitney.loc[mann_whitney["Metric"].eq("PageValues")].iloc[0]

    strongest_early = early_behavior.iloc[
        early_behavior["RankBiserialCorrelation"].abs().argmax()
    ]

    lines = [
        "# 电商网站会话购买转化统计检验报告",
        "",
        "## 1. 方法与口径",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 输入文件：`{input_path.name}`",
        f"- 输入文件 SHA-256：`{sha256(input_path)}`",
        f"- 样本量：{len(data):,} 个会话",
        f"- 显著性水平：{ALPHA:.2f}",
        "- 类别变量：Pearson 卡方检验，不使用 Yates 连续性校正；效应量为 Cramér's V。",
        "- 连续/有序行为变量：双侧渐近 Mann-Whitney U 检验，处理并列秩并使用连续性校正；效应量为秩二列相关系数。",
        "- 多重检验：行为指标组内和访客类型两两比较组内分别使用 Benjamini-Hochberg 校正。",
        "- 方向定义：秩二列相关系数为正表示购买会话数值倾向更高，为负表示购买会话数值倾向更低。",
        "- 效应量解释：绝对值小于0.10为可忽略，0.10至0.30为小，0.30至0.50为中，0.50及以上为大。",
        "",
        "## 2. 卡方检验汇总",
        "",
    ]
    chi_rows = [
        [
            row["Dimension"],
            number(row["ChiSquare"]),
            int(row["DegreesOfFreedom"]),
            format_p(row["PValue"]),
            format_p(row["AdjustedPValueBH"]),
            number(row["CramerV"]),
            row["EffectMagnitude"],
            number(row["MinimumExpectedCount"], 2),
        ]
        for _, row in chi_square.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["关系", "χ²", "df", "p值", "BH校正p值", "Cramér's V", "效应", "最小期望频数"],
            chi_rows,
        )
    )
    lines.extend(
        [
            "",
            "两个检验的所有期望频数均不小于5，卡方检验的常用频数条件得到满足。统计显著只表示变量之间存在关联，不表示访客类型或周末访问导致购买。",
            "",
            "## 3. 访客类型与购买",
            "",
            f"总体检验结果为 χ²({int(visitor_test['DegreesOfFreedom'])})={number(visitor_test['ChiSquare'])}，p={format_p(visitor_test['PValue'])}，Cramér's V={number(visitor_test['CramerV'])}（{visitor_test['EffectMagnitude']}效应）。",
            "",
        ]
    )
    visitor_rows = [
        [
            row["Category"],
            f"{int(row['TotalSessions']):,}",
            f"{int(row['PurchaseSessions']):,}",
            pct(row["ConversionRate"]),
            f"{pct(row['ConversionCILower95'])}至{pct(row['ConversionCIUpper95'])}",
            number(row["PearsonResidualPurchase"]),
        ]
        for _, row in visitor_detail.sort_values("TotalSessions", ascending=False).iterrows()
    ]
    lines.extend(
        markdown_table(
            ["访客类型", "会话数", "购买会话", "转化率", "95%置信区间", "购买单元格Pearson残差"],
            visitor_rows,
        )
    )
    lines.extend(
        [
            "",
            "正的购买单元格残差表示实际购买会话数高于独立性假设下的期望数量，负值表示低于期望。",
            "",
            "### 3.1 访客类型两两比较",
            "",
        ]
    )
    pairwise_rows = [
        [
            f"{row['CategoryA']} vs {row['CategoryB']}",
            f"{pct(row['ConversionRateA'])} vs {pct(row['ConversionRateB'])}",
            f"{row['ConversionDifferenceAminusB'] * 100:+.2f}个百分点",
            number(row["OddsRatioAtoB"]),
            format_p(row["AdjustedPValueBH"]),
            number(row["CramerV"]),
            "显著" if row["SignificantAfterBH"] else "不显著",
        ]
        for _, row in pairwise.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["比较", "转化率", "差值", "优势比A/B", "BH校正p值", "Cramér's V", "结果"],
            pairwise_rows,
        )
    )

    lines.extend(
        [
            "",
            "## 4. 周末与购买",
            "",
            f"周末与购买的检验结果为 χ²({int(weekend_test['DegreesOfFreedom'])})={number(weekend_test['ChiSquare'])}，p={format_p(weekend_test['PValue'])}，Cramér's V={number(weekend_test['CramerV'])}（{weekend_test['EffectMagnitude']}效应）。",
            "",
        ]
    )
    weekend_rows = [
        [
            row["Category"],
            f"{int(row['TotalSessions']):,}",
            f"{int(row['PurchaseSessions']):,}",
            pct(row["ConversionRate"]),
            f"{pct(row['ConversionCILower95'])}至{pct(row['ConversionCIUpper95'])}",
        ]
        for _, row in weekend_detail.sort_values("Category").iterrows()
    ]
    lines.extend(
        markdown_table(
            ["日期类型", "会话数", "购买会话", "转化率", "95%置信区间"], weekend_rows
        )
    )
    lines.extend(
        [
            "",
            f"周末转化率比工作日高 {weekend_effect['ConversionDifferenceWeekendMinusWeekday'] * 100:.2f} 个百分点，相对风险为 {number(weekend_effect['RelativeRiskWeekendToWeekday'])}，优势比为 {number(weekend_effect['OddsRatioWeekendToWeekday'])}。虽然差异达到统计显著，但 Cramér's V 表明关联强度{weekend_test['EffectMagnitude']}，业务解释应保持克制。",
            "",
            "## 5. 购买与未购买会话行为差异",
            "",
            "下表按效应量绝对值从高到低排列。所有校正均在本节的9项行为指标内完成。",
            "",
        ]
    )
    mann_rows = [
        [
            row["MetricName"],
            f"{number(row['PurchaseMedian'])} [{number(row['PurchaseQ1'])}, {number(row['PurchaseQ3'])}]",
            f"{number(row['NonPurchaseMedian'])} [{number(row['NonPurchaseQ1'])}, {number(row['NonPurchaseQ3'])}]",
            number(row["UStatistic"], 0),
            format_p(row["AdjustedPValueBH"]),
            number(row["RankBiserialCorrelation"]),
            f"{row['EffectDirection']}，{row['EffectMagnitude']}效应",
        ]
        for _, row in mann_whitney.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["指标", "购买中位数 [Q1,Q3]", "未购买中位数 [Q1,Q3]", "U统计量", "BH校正p值", "秩二列相关", "方向与效应"],
            mann_rows,
        )
    )

    significant_count = int(mann_whitney["SignificantAfterBH"].sum())
    lines.extend(
        [
            "",
            f"9项行为指标中有 {significant_count} 项在BH校正后仍达到0.05显著性水平。排除 `PageValues` 后，绝对效应量最大的是 `{strongest_early['Metric']}`，秩二列相关为 {number(strongest_early['RankBiserialCorrelation'])}（{strongest_early['EffectMagnitude']}效应）。",
            "",
            "## 6. PageValues专项解释",
            "",
            f"`PageValues` 的购买会话中位数为 {number(page_values['PurchaseMedian'])}，未购买会话中位数为 {number(page_values['NonPurchaseMedian'])}；秩二列相关为 {number(page_values['RankBiserialCorrelation'])}（{page_values['EffectMagnitude']}效应），BH校正p值为 {format_p(page_values['AdjustedPValueBH'])}。",
            "",
            "该结果表明 `PageValues` 对购买结果具有很强的区分能力，但不能据此认定它适合早期干预。该字段可能在临近交易时才形成，后续必须分别建立包含它的会话末期模型和排除它的早期倾向模型。",
            "",
            "## 7. 业务结论",
            "",
            "1. 访客类型与购买结果存在统计关联，但效应量决定了它不能单独作为运营分群依据；应与页面行为和流量来源联合使用。",
            "2. 周末会话的样本转化率高于工作日，但关联强度很弱，不能仅据此大幅调整周末资源配置。",
            "3. 产品页浏览量、产品页停留时长、跳出率和退出率的差异在统计上得到支持；效应方向与EDA一致。",
            "4. 跳出率和退出率为负向效应，说明购买会话整体倾向于更低的跳出与退出水平，但仍不能解释因果机制。",
            "5. `PageValues` 的效应远强于多数早期行为变量，应单独管理其使用时点，避免部署场景中的信息泄漏。",
            "",
            "## 8. 限制",
            "",
            "- 大样本会使较小差异也可能达到统计显著，因此判断优先级时必须同时阅读效应量。",
            "- Mann-Whitney U检验比较的是两组总体分布和随机优势，不应简化为只检验中位数。",
            "- 当前检验没有控制月份、流量来源和访客类型之间的混杂关系；多变量影响将在逻辑回归和树模型阶段处理。",
            "- 结果是会话级关联，不支持用户级留存、复购或因果结论。",
            "",
            "## 9. 输出文件",
            "",
            "结果位于 `outputs/statistics/`：",
            "",
            "- `chi_square_tests.csv`",
            "- `categorical_contingency_details.csv`",
            "- `visitor_type_pairwise_tests.csv`",
            "- `weekend_effects.csv`",
            "- `mann_whitney_tests.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def validate_results(
    chi_square: pd.DataFrame,
    pairwise: pd.DataFrame,
    mann_whitney: pd.DataFrame,
) -> None:
    if len(chi_square) != 2:
        raise AssertionError("Expected two chi-square tests.")
    if len(pairwise) != 3:
        raise AssertionError("Expected three visitor pairwise tests.")
    if len(mann_whitney) != len(BEHAVIOR_METRICS):
        raise AssertionError("Unexpected number of Mann-Whitney tests.")
    for table in [chi_square, pairwise, mann_whitney]:
        if table.isna().any().any():
            raise AssertionError("A statistical result table contains missing values.")
    for table in [chi_square, pairwise, mann_whitney]:
        if "AdjustedPValueBH" in table:
            if (table["AdjustedPValueBH"] + 1e-15 < table["PValue"]).any():
                raise AssertionError("A BH-adjusted p-value is below its raw p-value.")
    if not chi_square["AssumptionMet"].all():
        raise AssertionError("A chi-square expected-frequency assumption is not met.")
    if not mann_whitney["RankBiserialCorrelation"].between(-1, 1).all():
        raise AssertionError("A rank-biserial correlation is outside [-1, 1].")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)

    chi_summaries = []
    contingency_tables = []
    for source_column, dimension_name in [
        ("VisitorType", "访客类型与购买"),
        ("WeekendLabel", "周末与购买"),
    ]:
        summary, details = chi_square_test(data, source_column, dimension_name)
        chi_summaries.append(summary)
        contingency_tables.append(details)

    chi_square = pd.DataFrame(chi_summaries)
    chi_square["AdjustedPValueBH"] = bh_adjust(chi_square["PValue"])
    chi_square["SignificantAfterBH"] = chi_square["AdjustedPValueBH"].lt(ALPHA)
    contingency = pd.concat(contingency_tables, ignore_index=True)
    pairwise = visitor_pairwise_tests(data)
    weekend = weekend_effects(data)
    mann_whitney = mann_whitney_tests(data)

    validate_results(chi_square, pairwise, mann_whitney)

    outputs = {
        "chi_square_tests.csv": chi_square,
        "categorical_contingency_details.csv": contingency,
        "visitor_type_pairwise_tests.csv": pairwise,
        "weekend_effects.csv": weekend,
        "mann_whitney_tests.csv": mann_whitney,
    }
    for filename, table in outputs.items():
        write_csv(table, output_dir / filename)

    report = build_report(
        input_path,
        data,
        chi_square,
        contingency,
        pairwise,
        weekend,
        mann_whitney,
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows analyzed: {len(data):,}")
    print(f"Chi-square tests: {len(chi_square)}")
    print(f"Visitor pairwise tests: {len(pairwise)}")
    print(f"Mann-Whitney tests: {len(mann_whitney)}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
