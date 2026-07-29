"""Run descriptive exploratory analysis for session purchase conversion.

This stage describes observed differences only. It does not perform hypothesis
tests, causal inference, or predictive modeling.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "SessionID",
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "Month",
    "MonthOrder",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
    "RevenueFlag",
    "ProductDepthGroup",
    "ProductDepthOrder",
    "ProductDurationGroup",
    "ProductDurationOrder",
    "HighBounceSession",
    "HighExitSession",
}

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
        default=project_dir / "outputs" / "eda",
        help="Directory for EDA result tables.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "04_探索性分析报告.md",
        help="Path for the Markdown EDA report.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    invalid = sorted(set(normalized) - {"TRUE", "FALSE"})
    if invalid:
        raise ValueError(f"Unexpected values in {column}: {invalid}")
    return normalized.eq("TRUE")


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("The analysis dataset contains no rows.")
    if data["SessionID"].duplicated().any():
        raise ValueError("SessionID is not unique.")

    for column in ["Weekend", "Revenue", "HighBounceSession", "HighExitSession"]:
        data[column] = as_bool(data[column], column)

    expected_revenue = data["Revenue"].astype(int)
    if not data["RevenueFlag"].astype(int).equals(expected_revenue):
        raise ValueError("RevenueFlag does not match Revenue.")
    data["RevenueFlag"] = data["RevenueFlag"].astype(int)
    return data


def grouped_performance(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        data.groupby(group_columns, observed=True, dropna=False)
        .agg(
            Sessions=("SessionID", "count"),
            Purchases=("RevenueFlag", "sum"),
            AvgProductPages=("ProductRelated", "mean"),
            MedianProductPages=("ProductRelated", "median"),
            AvgProductDuration=("ProductRelated_Duration", "mean"),
            MedianProductDuration=("ProductRelated_Duration", "median"),
            AvgBounceRate=("BounceRates", "mean"),
            AvgExitRate=("ExitRates", "mean"),
        )
        .reset_index()
    )
    grouped["ConversionRate"] = grouped["Purchases"] / grouped["Sessions"]
    grouped["SessionShare"] = grouped["Sessions"] / len(data)
    return grouped


def behavior_comparison(data: pd.DataFrame) -> pd.DataFrame:
    purchased = data.loc[data["RevenueFlag"].eq(1)]
    not_purchased = data.loc[data["RevenueFlag"].eq(0)]
    rows = []
    for metric in BEHAVIOR_METRICS:
        buy = purchased[metric]
        no_buy = not_purchased[metric]
        rows.append(
            {
                "Metric": metric,
                "PurchaseMean": buy.mean(),
                "PurchaseMedian": buy.median(),
                "PurchaseP25": buy.quantile(0.25),
                "PurchaseP75": buy.quantile(0.75),
                "NonPurchaseMean": no_buy.mean(),
                "NonPurchaseMedian": no_buy.median(),
                "NonPurchaseP25": no_buy.quantile(0.25),
                "NonPurchaseP75": no_buy.quantile(0.75),
                "MedianDifference": buy.median() - no_buy.median(),
            }
        )
    return pd.DataFrame(rows)


def page_presence_performance(data: pd.DataFrame) -> pd.DataFrame:
    tables = []
    definitions = {
        "Administrative": "管理类页面",
        "Informational": "信息类页面",
        "ProductRelated": "产品相关页面",
    }
    for column, label in definitions.items():
        working = data.assign(PageVisitStatus=data[column].gt(0).map({True: "访问", False: "未访问"}))
        table = grouped_performance(working, ["PageVisitStatus"])
        table.insert(0, "PageType", label)
        table.insert(1, "SourceField", column)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def bounce_exit_performance(data: pd.DataFrame) -> pd.DataFrame:
    tables = []
    definitions = [
        ("HighBounceSession", "跳出率", "高跳出", "非高跳出"),
        ("HighExitSession", "退出率", "高退出", "非高退出"),
    ]
    for column, dimension, true_label, false_label in definitions:
        working = data.assign(
            Segment=data[column].map({True: true_label, False: false_label})
        )
        table = grouped_performance(working, ["Segment"])
        table.insert(0, "Dimension", dimension)
        table.insert(1, "SourceField", column)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def build_opportunity_segments(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    not_purchased = data["RevenueFlag"].eq(0)
    low_bounce = ~data["HighBounceSession"]
    low_exit = ~data["HighExitSession"]
    deep_browse = data["ProductDepthOrder"].ge(4)
    high_duration = data["ProductDurationOrder"].eq(3)
    returning = data["VisitorType"].eq("Returning_Visitor")

    segments = OrderedDict(
        [
            (
                "高浏览未购买",
                (
                    not_purchased & data["ProductDepthGroup"].eq("50及以上"),
                    "未购买且产品页浏览量不少于50",
                ),
            ),
            (
                "长停留未购买",
                (
                    not_purchased & high_duration,
                    "未购买且产品页停留时长位于高分位组",
                ),
            ),
            (
                "回访未购买",
                (
                    not_purchased & returning,
                    "未购买且访客类型为Returning_Visitor",
                ),
            ),
            (
                "深度浏览且低跳出低退出",
                (
                    not_purchased & deep_browse & low_bounce & low_exit,
                    "未购买、产品页浏览量不少于21且非高跳出/高退出",
                ),
            ),
            (
                "高潜组合会话",
                (
                    not_purchased
                    & returning
                    & deep_browse
                    & high_duration
                    & low_bounce
                    & low_exit,
                    "未购买、回访、深度浏览、长停留且非高跳出/高退出",
                ),
            ),
        ]
    )

    unpurchased_count = int(not_purchased.sum())
    rows = []
    high_potential_mask = None
    for segment, (mask, definition) in segments.items():
        subset = data.loc[mask]
        rows.append(
            {
                "Segment": segment,
                "Definition": definition,
                "Sessions": len(subset),
                "ShareOfUnpurchased": len(subset) / unpurchased_count,
                "AvgProductPages": subset["ProductRelated"].mean(),
                "MedianProductPages": subset["ProductRelated"].median(),
                "AvgProductDuration": subset["ProductRelated_Duration"].mean(),
                "MedianProductDuration": subset["ProductRelated_Duration"].median(),
                "AvgBounceRate": subset["BounceRates"].mean(),
                "AvgExitRate": subset["ExitRates"].mean(),
            }
        )
        if segment == "高潜组合会话":
            high_potential_mask = mask

    if high_potential_mask is None:
        raise AssertionError("High-potential segment was not created.")

    detail_columns = [
        "SessionID",
        "RevenueFlag",
        "VisitorType",
        "TrafficType",
        "Month",
        "Weekend",
        "ProductRelated",
        "ProductRelated_Duration",
        "ProductDepthGroup",
        "ProductDurationGroup",
        "BounceRates",
        "ExitRates",
        "HighBounceSession",
        "HighExitSession",
    ]
    high_potential = (
        data.loc[high_potential_mask, detail_columns]
        .sort_values(
            ["ProductRelated", "ProductRelated_Duration"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    return pd.DataFrame(rows), high_potential


def traffic_opportunity_table(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    overall_conversion = data["RevenueFlag"].mean()
    table = grouped_performance(data, ["TrafficType"]).sort_values("TrafficType")
    volume_threshold = float(table["Sessions"].median())
    table["VolumeLevel"] = table["Sessions"].ge(volume_threshold).map(
        {True: "高流量", False: "低流量"}
    )
    table["ConversionLevel"] = table["ConversionRate"].ge(overall_conversion).map(
        {True: "高转化", False: "低转化"}
    )
    table["OpportunityQuadrant"] = table["VolumeLevel"] + table["ConversionLevel"]
    table["ExpectedPurchasesAtOverallRate"] = table["Sessions"] * overall_conversion
    table["PurchaseGapToOverall"] = (
        table["ExpectedPurchasesAtOverallRate"] - table["Purchases"]
    ).clip(lower=0)
    table["SampleAtLeast100"] = table["Sessions"].ge(100)
    return table, volume_threshold


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8f")


def pct(value: float) -> str:
    return f"{value:.2%}"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def performance_markdown(
    table: pd.DataFrame, label_column: str, limit: int | None = None
) -> list[str]:
    shown = table if limit is None else table.head(limit)
    rows = [
        [
            row[label_column],
            f"{int(row['Sessions']):,}",
            f"{int(row['Purchases']):,}",
            pct(row["ConversionRate"]),
            pct(row["SessionShare"]),
        ]
        for _, row in shown.iterrows()
    ]
    return markdown_table(
        [label_column, "会话数", "购买会话数", "转化率", "会话占比"], rows
    )


def build_report(
    input_path: Path,
    data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    traffic_volume_threshold: float,
) -> str:
    sessions = len(data)
    purchases = int(data["RevenueFlag"].sum())
    conversion = purchases / sessions
    visitor = tables["visitor_type"]
    month = tables["month"]
    weekend = tables["weekend"]
    traffic = tables["traffic_type"]
    depth = tables["product_depth"]
    duration = tables["product_duration"]
    bounce_exit = tables["bounce_exit"]
    behavior = tables["behavior_comparison"].set_index("Metric")
    opportunities = tables["opportunity_segments"]
    high_potential = tables["high_potential_sessions"]

    new_row = visitor.loc[visitor["VisitorType"].eq("New_Visitor")].iloc[0]
    returning_row = visitor.loc[
        visitor["VisitorType"].eq("Returning_Visitor")
    ].iloc[0]
    weekend_row = weekend.loc[weekend["DayType"].eq("周末")].iloc[0]
    weekday_row = weekend.loc[weekend["DayType"].eq("工作日")].iloc[0]
    best_month = month.loc[month["ConversionRate"].idxmax()]
    lowest_month = month.loc[month["ConversionRate"].idxmin()]

    reliable_traffic = traffic.loc[traffic["SampleAtLeast100"]]
    best_traffic = reliable_traffic.loc[reliable_traffic["ConversionRate"].idxmax()]
    low_conversion_large = reliable_traffic.loc[
        reliable_traffic["ConversionRate"].lt(conversion)
    ].sort_values(["Sessions", "PurchaseGapToOverall"], ascending=[False, False])

    low_depth = depth.sort_values("ProductDepthOrder").iloc[0]
    high_depth = depth.sort_values("ProductDepthOrder").iloc[-1]
    low_duration = duration.sort_values("ProductDurationOrder").iloc[0]
    high_duration = duration.sort_values("ProductDurationOrder").iloc[-1]
    high_bounce = bounce_exit.loc[bounce_exit["Segment"].eq("高跳出")].iloc[0]
    normal_bounce = bounce_exit.loc[bounce_exit["Segment"].eq("非高跳出")].iloc[0]
    high_exit = bounce_exit.loc[bounce_exit["Segment"].eq("高退出")].iloc[0]
    normal_exit = bounce_exit.loc[bounce_exit["Segment"].eq("非高退出")].iloc[0]

    page_value = behavior.loc["PageValues"]
    product_pages = behavior.loc["ProductRelated"]
    product_duration = behavior.loc["ProductRelated_Duration"]
    bounce_metric = behavior.loc["BounceRates"]
    exit_metric = behavior.loc["ExitRates"]

    lines = [
        "# 电商网站会话购买转化探索性分析报告",
        "",
        "## 1. 分析说明",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 输入文件：`{input_path.name}`",
        f"- 输入文件 SHA-256：`{sha256(input_path)}`",
        f"- 数据规模：{sessions:,} 个会话，{data.shape[1]} 个字段",
        "- 分析性质：描述性探索分析，不包含显著性检验、因果推断或预测建模。",
        "- 注意：各分组转化率必须结合会话量阅读；渠道编号暂无业务名称映射。",
        "",
        "## 2. 整体表现",
        "",
    ]
    overall_rows = [
        ["会话数量", f"{sessions:,}"],
        ["购买会话数", f"{purchases:,}"],
        ["会话购买转化率", pct(conversion)],
        ["新访客会话占比", pct(new_row["Sessions"] / sessions)],
        ["回访访客会话转化率", pct(returning_row["ConversionRate"])],
        ["平均产品页浏览量", num(data["ProductRelated"].mean())],
        ["产品页浏览量中位数", num(data["ProductRelated"].median(), 0)],
    ]
    lines.extend(markdown_table(["指标", "结果"], overall_rows))

    lines.extend(
        [
            "",
            "## 3. 访客类型",
            "",
        ]
    )
    visitor_display = visitor.sort_values("Sessions", ascending=False)
    lines.extend(performance_markdown(visitor_display, "VisitorType"))
    lines.extend(
        [
            "",
            f"新访客会话转化率为 {pct(new_row['ConversionRate'])}，高于回访访客会话的 {pct(returning_row['ConversionRate'])}；但回访访客贡献了 {int(returning_row['Sessions']):,} 个会话，是主要流量主体。该差异仍需在后续阶段用卡方检验和效应量验证。",
            "",
            "## 4. 周末与月份",
            "",
            f"周末会话转化率为 {pct(weekend_row['ConversionRate'])}，工作日为 {pct(weekday_row['ConversionRate'])}。这是样本差异，尚未控制月份、访客类型或流量结构。",
            "",
            f"样本月份中，转化率最高的是 {best_month['Month']}（{pct(best_month['ConversionRate'])}，{int(best_month['Sessions']):,} 个会话），最低的是 {lowest_month['Month']}（{pct(lowest_month['ConversionRate'])}，{int(lowest_month['Sessions']):,} 个会话）。由于没有年份和完整12个月数据，不将该结果表述为长期季节性。",
            "",
        ]
    )
    lines.extend(performance_markdown(month.sort_values("MonthOrder"), "Month"))

    lines.extend(["", "## 5. 流量来源", ""])
    traffic_display = traffic.sort_values("Sessions", ascending=False)
    lines.extend(performance_markdown(traffic_display, "TrafficType"))
    lines.extend(
        [
            "",
            f"在会话数不少于100的来源中，TrafficType {int(best_traffic['TrafficType'])} 的转化率最高，为 {pct(best_traffic['ConversionRate'])}，对应 {int(best_traffic['Sessions']):,} 个会话。",
            f"流量高低以各 TrafficType 会话数中位数 {traffic_volume_threshold:,.0f} 为描述性分界；转化高低以整体转化率 {pct(conversion)} 为分界。该四象限仅用于筛选线索，不代表渠道ROI。",
            "",
            "优先核查的高流量低转化来源：",
            "",
        ]
    )
    low_traffic_rows = [
        [
            int(row["TrafficType"]),
            f"{int(row['Sessions']):,}",
            pct(row["ConversionRate"]),
            num(row["PurchaseGapToOverall"], 1),
        ]
        for _, row in low_conversion_large.head(5).iterrows()
    ]
    lines.extend(
        markdown_table(
            ["TrafficType", "会话数", "转化率", "达到整体转化率的购买差额"],
            low_traffic_rows,
        )
    )

    lines.extend(["", "## 6. 产品页行为", ""])
    lines.extend(
        [
            f"购买会话的产品页浏览量中位数为 {num(product_pages['PurchaseMedian'], 0)}，未购买会话为 {num(product_pages['NonPurchaseMedian'], 0)}；产品页停留时间中位数分别为 {num(product_duration['PurchaseMedian'])} 和 {num(product_duration['NonPurchaseMedian'])}。",
            "",
            f"浏览深度从“{low_depth['ProductDepthGroup']}”到“{high_depth['ProductDepthGroup']}”时，观察到的会话转化率从 {pct(low_depth['ConversionRate'])} 变为 {pct(high_depth['ConversionRate'])}。",
            f"停留时长从“{low_duration['ProductDurationGroup']}”到“{high_duration['ProductDurationGroup']}”时，观察到的会话转化率从 {pct(low_duration['ConversionRate'])} 变为 {pct(high_duration['ConversionRate'])}。这些是关联关系，不代表增加浏览或停留一定会导致购买。",
            "",
            "### 6.1 产品页浏览深度",
            "",
        ]
    )
    lines.extend(performance_markdown(depth.sort_values("ProductDepthOrder"), "ProductDepthGroup"))
    lines.extend(["", "### 6.2 产品页停留时长", ""])
    lines.extend(
        performance_markdown(
            duration.sort_values("ProductDurationOrder"), "ProductDurationGroup"
        )
    )

    lines.extend(["", "## 7. 跳出率与退出率", ""])
    lines.extend(
        [
            f"高跳出会话转化率为 {pct(high_bounce['ConversionRate'])}，非高跳出会话为 {pct(normal_bounce['ConversionRate'])}；高退出会话转化率为 {pct(high_exit['ConversionRate'])}，非高退出会话为 {pct(normal_exit['ConversionRate'])}。",
            "",
            f"从购买结果看，购买会话的 BounceRates 中位数为 {num(bounce_metric['PurchaseMedian'], 4)}，未购买会话为 {num(bounce_metric['NonPurchaseMedian'], 4)}；ExitRates 中位数分别为 {num(exit_metric['PurchaseMedian'], 4)} 和 {num(exit_metric['NonPurchaseMedian'], 4)}。",
            "",
        ]
    )
    bounce_rows = [
        [
            row["Segment"],
            f"{int(row['Sessions']):,}",
            pct(row["ConversionRate"]),
            num(row["AvgProductPages"]),
            num(row["MedianProductDuration"]),
        ]
        for _, row in bounce_exit.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["会话类型", "会话数", "转化率", "平均产品页数", "产品页停留中位数"],
            bounce_rows,
        )
    )

    lines.extend(["", "## 8. 购买与未购买行为差异", ""])
    comparison_rows = []
    for metric in BEHAVIOR_METRICS:
        row = behavior.loc[metric]
        comparison_rows.append(
            [
                metric,
                num(row["PurchaseMedian"], 4),
                num(row["NonPurchaseMedian"], 4),
                num(row["PurchaseMean"], 4),
                num(row["NonPurchaseMean"], 4),
            ]
        )
    lines.extend(
        markdown_table(
            ["指标", "购买中位数", "未购买中位数", "购买均值", "未购买均值"],
            comparison_rows,
        )
    )
    lines.extend(
        [
            "",
            f"`PageValues` 的购买会话中位数为 {num(page_value['PurchaseMedian'], 4)}，未购买会话中位数为 {num(page_value['NonPurchaseMedian'], 4)}，区分度很强。该字段可能只在临近购买时可用，不能直接作为早期运营干预依据。",
            "",
            "## 9. 初步增长机会",
            "",
            "以下会话类型可以相互重叠，数量不能直接相加。高潜组合定义排除了 `PageValues`，用于模拟较早阶段可观察的行为线索。",
            "",
        ]
    )
    opportunity_rows = [
        [
            row["Segment"],
            f"{int(row['Sessions']):,}",
            pct(row["ShareOfUnpurchased"]),
            num(row["MedianProductPages"], 0),
            num(row["MedianProductDuration"]),
            pct(row["AvgExitRate"]),
        ]
        for _, row in opportunities.iterrows()
    ]
    lines.extend(
        markdown_table(
            ["机会会话类型", "会话数", "未购买会话占比", "产品页中位数", "停留中位数", "平均退出率"],
            opportunity_rows,
        )
    )
    top_high_potential_traffic = (
        high_potential.groupby("TrafficType", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(3)
    )
    traffic_text = "、".join(
        f"TrafficType {int(index)}（{int(count):,}条）"
        for index, count in top_high_potential_traffic.items()
    )
    lines.extend(
        [
            "",
            f"高潜组合会话共有 {len(high_potential):,} 条，最集中的三个流量来源是 {traffic_text}。这些会话适合在后续模型和业务规则中优先验证。",
            "",
            "## 10. 本阶段建议",
            "",
            "1. 渠道侧先核查高流量低转化的 TrafficType，结合渠道映射和获客成本后再决定预算调整。",
            "2. 产品侧优先研究深度浏览、长停留但未购买的回访会话，排查商品信息、信任要素、价格比较和结算入口等阻碍。",
            "3. 高跳出和高退出会话应按来源、访客类型和产品浏览深度继续拆分，避免把所有高退出简单归因于页面体验。",
            "4. `PageValues` 单独用于会话末期分析；早期倾向模型和早期干预规则排除该字段。",
            "5. 下一阶段使用卡方检验、Mann-Whitney U 检验和效应量判断观察差异是否具有统计与业务意义。",
            "",
            "## 11. 输出表",
            "",
            "所有明细结果位于 `outputs/eda/`：",
            "",
            "- `overall_kpis.csv`",
            "- `visitor_type_performance.csv`",
            "- `month_performance.csv`",
            "- `weekend_performance.csv`",
            "- `traffic_type_performance.csv`",
            "- `operating_system_performance.csv`",
            "- `browser_performance.csv`",
            "- `region_performance.csv`",
            "- `product_depth_performance.csv`",
            "- `product_duration_performance.csv`",
            "- `page_presence_performance.csv`",
            "- `bounce_exit_performance.csv`",
            "- `purchase_behavior_comparison.csv`",
            "- `growth_opportunity_segments.csv`",
            "- `high_potential_unpurchased_sessions.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)
    overall_conversion = data["RevenueFlag"].mean()

    overall_kpis = pd.DataFrame(
        [
            ("Sessions", len(data), f"{len(data):,}"),
            ("Purchases", int(data["RevenueFlag"].sum()), f"{int(data['RevenueFlag'].sum()):,}"),
            ("ConversionRate", overall_conversion, pct(overall_conversion)),
            (
                "NewVisitorShare",
                data["VisitorType"].eq("New_Visitor").mean(),
                pct(data["VisitorType"].eq("New_Visitor").mean()),
            ),
            (
                "ReturningVisitorConversionRate",
                data.loc[data["VisitorType"].eq("Returning_Visitor"), "RevenueFlag"].mean(),
                pct(data.loc[data["VisitorType"].eq("Returning_Visitor"), "RevenueFlag"].mean()),
            ),
            ("AvgProductPages", data["ProductRelated"].mean(), num(data["ProductRelated"].mean())),
            ("MedianProductPages", data["ProductRelated"].median(), num(data["ProductRelated"].median(), 0)),
        ],
        columns=["Metric", "Value", "FormattedValue"],
    )

    visitor_type = grouped_performance(data, ["VisitorType"])
    month = grouped_performance(data, ["Month", "MonthOrder"]).sort_values("MonthOrder")

    weekend_working = data.assign(DayType=data["Weekend"].map({True: "周末", False: "工作日"}))
    weekend = grouped_performance(weekend_working, ["DayType"])
    weekend["DayTypeOrder"] = weekend["DayType"].map({"工作日": 1, "周末": 2})
    weekend = weekend.sort_values("DayTypeOrder")

    traffic_type, traffic_volume_threshold = traffic_opportunity_table(data)
    operating_system = grouped_performance(data, ["OperatingSystems"]).sort_values("OperatingSystems")
    operating_system["SampleAtLeast100"] = operating_system["Sessions"].ge(100)
    browser = grouped_performance(data, ["Browser"]).sort_values("Browser")
    browser["SampleAtLeast100"] = browser["Sessions"].ge(100)
    region = grouped_performance(data, ["Region"]).sort_values("Region")
    region["SampleAtLeast100"] = region["Sessions"].ge(100)

    product_depth = grouped_performance(
        data, ["ProductDepthGroup", "ProductDepthOrder"]
    ).sort_values("ProductDepthOrder")
    product_duration = grouped_performance(
        data, ["ProductDurationGroup", "ProductDurationOrder"]
    ).sort_values("ProductDurationOrder")
    page_presence = page_presence_performance(data)
    bounce_exit = bounce_exit_performance(data)
    purchase_comparison = behavior_comparison(data)
    opportunity_segments, high_potential = build_opportunity_segments(data)

    tables = {
        "overall_kpis": overall_kpis,
        "visitor_type": visitor_type,
        "month": month,
        "weekend": weekend,
        "traffic_type": traffic_type,
        "operating_system": operating_system,
        "browser": browser,
        "region": region,
        "product_depth": product_depth,
        "product_duration": product_duration,
        "page_presence": page_presence,
        "bounce_exit": bounce_exit,
        "behavior_comparison": purchase_comparison,
        "opportunity_segments": opportunity_segments,
        "high_potential_sessions": high_potential,
    }

    filenames = {
        "overall_kpis": "overall_kpis.csv",
        "visitor_type": "visitor_type_performance.csv",
        "month": "month_performance.csv",
        "weekend": "weekend_performance.csv",
        "traffic_type": "traffic_type_performance.csv",
        "operating_system": "operating_system_performance.csv",
        "browser": "browser_performance.csv",
        "region": "region_performance.csv",
        "product_depth": "product_depth_performance.csv",
        "product_duration": "product_duration_performance.csv",
        "page_presence": "page_presence_performance.csv",
        "bounce_exit": "bounce_exit_performance.csv",
        "behavior_comparison": "purchase_behavior_comparison.csv",
        "opportunity_segments": "growth_opportunity_segments.csv",
        "high_potential_sessions": "high_potential_unpurchased_sessions.csv",
    }
    for name, filename in filenames.items():
        write_csv(tables[name], output_dir / filename)

    report = build_report(
        input_path, data, tables, traffic_volume_threshold
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows analyzed: {len(data):,}")
    print(f"Overall conversion: {overall_conversion:.2%}")
    print(f"Tables written: {len(filenames)}")
    print(f"High-potential unpurchased sessions: {len(high_potential):,}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
