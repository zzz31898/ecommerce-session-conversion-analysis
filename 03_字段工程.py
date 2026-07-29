"""Create analysis-ready fields for session conversion analysis.

The script preserves every source row and source field. Quantile thresholds
are recalculated from the input data on every run and documented in a report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


SOURCE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

DERIVED_COLUMNS = [
    "SessionID",
    "RevenueFlag",
    "MonthOrder",
    "ProductDepthGroup",
    "ProductDepthOrder",
    "ProductDurationGroup",
    "ProductDurationOrder",
    "HighBounceSession",
    "HighExitSession",
]

OUTPUT_COLUMNS = ["SessionID", *SOURCE_COLUMNS, *DERIVED_COLUMNS[1:]]

MONTH_ORDER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

PRODUCT_DEPTH_ORDER = {
    "0至5": 1,
    "6至10": 2,
    "11至20": 3,
    "21至49": 4,
    "50及以上": 5,
}

PRODUCT_DURATION_ORDER = {"低": 1, "中": 2, "高": 3}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "data" / "processed" / "online_shoppers_cleaned_base.csv",
        help="Path to the cleaned base CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "online_shoppers_analysis_ready_v2.csv",
        help="Path for the analysis-ready CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "03_派生字段说明.md",
        help="Path for the derived-field report.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile from an empty list.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fmt_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.9f}".rstrip("0").rstrip(".")


def fmt_percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "NA"


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def read_source(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SOURCE_COLUMNS:
            raise ValueError(
                "Unexpected input columns.\n"
                f"Expected: {SOURCE_COLUMNS}\n"
                f"Actual:   {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("The input CSV contains no data rows.")
    return rows


def product_depth_group(value: int) -> tuple[str, int]:
    if value < 0:
        raise ValueError(f"ProductRelated cannot be negative: {value}")
    if value <= 5:
        label = "0至5"
    elif value <= 10:
        label = "6至10"
    elif value <= 20:
        label = "11至20"
    elif value < 50:
        label = "21至49"
    else:
        label = "50及以上"
    return label, PRODUCT_DEPTH_ORDER[label]


def product_duration_group(
    value: float, lower_threshold: float, upper_threshold: float
) -> tuple[str, int]:
    if value <= lower_threshold:
        label = "低"
    elif value <= upper_threshold:
        label = "中"
    else:
        label = "高"
    return label, PRODUCT_DURATION_ORDER[label]


def build_analysis_rows(
    source_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    product_durations = [float(row["ProductRelated_Duration"]) for row in source_rows]
    bounce_rates = [float(row["BounceRates"]) for row in source_rows]
    exit_rates = [float(row["ExitRates"]) for row in source_rows]

    thresholds = {
        "duration_p33": quantile(product_durations, 1 / 3),
        "duration_p67": quantile(product_durations, 2 / 3),
        "bounce_p75": quantile(bounce_rates, 0.75),
        "exit_p75": quantile(exit_rates, 0.75),
    }
    if thresholds["duration_p33"] >= thresholds["duration_p67"]:
        raise ValueError(
            "Product duration tertile thresholds are not distinct. "
            "Review zero values before defining duration groups."
        )

    analysis_rows: list[dict[str, object]] = []
    for session_id, source_row in enumerate(source_rows, start=1):
        month = source_row["Month"]
        if month not in MONTH_ORDER:
            raise ValueError(f"Unknown Month value: {month}")

        revenue = source_row["Revenue"]
        if revenue not in {"TRUE", "FALSE"}:
            raise ValueError(f"Unknown Revenue value: {revenue}")

        product_depth = int(source_row["ProductRelated"])
        depth_label, depth_order = product_depth_group(product_depth)

        product_duration = float(source_row["ProductRelated_Duration"])
        duration_label, duration_order = product_duration_group(
            product_duration,
            thresholds["duration_p33"],
            thresholds["duration_p67"],
        )

        bounce_rate = float(source_row["BounceRates"])
        exit_rate = float(source_row["ExitRates"])

        derived_values: dict[str, object] = {
            "SessionID": session_id,
            "RevenueFlag": 1 if revenue == "TRUE" else 0,
            "MonthOrder": MONTH_ORDER[month],
            "ProductDepthGroup": depth_label,
            "ProductDepthOrder": depth_order,
            "ProductDurationGroup": duration_label,
            "ProductDurationOrder": duration_order,
            "HighBounceSession": (
                "TRUE" if bounce_rate > thresholds["bounce_p75"] else "FALSE"
            ),
            "HighExitSession": (
                "TRUE" if exit_rate > thresholds["exit_p75"] else "FALSE"
            ),
        }
        analysis_rows.append({**source_row, **derived_values})

    return analysis_rows, thresholds


def write_output(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_output(
    source_rows: list[dict[str, str]], analysis_rows: list[dict[str, object]]
) -> None:
    if len(source_rows) != len(analysis_rows):
        raise AssertionError("Row count changed during field engineering.")

    session_ids = [int(row["SessionID"]) for row in analysis_rows]
    if session_ids != list(range(1, len(analysis_rows) + 1)):
        raise AssertionError("SessionID is not sequential and unique.")

    for source_row, analysis_row in zip(source_rows, analysis_rows):
        if any(source_row[column] != analysis_row[column] for column in SOURCE_COLUMNS):
            raise AssertionError("A source field changed during field engineering.")
        expected_revenue_flag = 1 if source_row["Revenue"] == "TRUE" else 0
        if analysis_row["RevenueFlag"] != expected_revenue_flag:
            raise AssertionError("RevenueFlag does not match Revenue.")
        if analysis_row["MonthOrder"] != MONTH_ORDER[source_row["Month"]]:
            raise AssertionError("MonthOrder does not match Month.")

    expected_labels = {
        "ProductDepthGroup": set(PRODUCT_DEPTH_ORDER),
        "ProductDurationGroup": set(PRODUCT_DURATION_ORDER),
        "HighBounceSession": {"TRUE", "FALSE"},
        "HighExitSession": {"TRUE", "FALSE"},
    }
    for column, valid_labels in expected_labels.items():
        actual_labels = {str(row[column]) for row in analysis_rows}
        if not actual_labels <= valid_labels:
            raise AssertionError(f"Unexpected labels in {column}: {actual_labels}")


def group_distribution(
    rows: list[dict[str, object]], column: str, ordered_values: list[str]
) -> list[tuple[str, str, str]]:
    counts = Counter(str(row[column]) for row in rows)
    return [
        (value, f"{counts[value]:,}", fmt_percent(counts[value], len(rows)))
        for value in ordered_values
    ]


def build_report(
    input_path: Path,
    output_path: Path,
    source_rows: list[dict[str, str]],
    analysis_rows: list[dict[str, object]],
    thresholds: dict[str, float],
) -> str:
    row_count = len(analysis_rows)
    zero_duration_count = sum(
        float(row["ProductRelated_Duration"]) == 0 for row in source_rows
    )

    field_rows = [
        ("SessionID", "整数", "从 1 开始的连续行索引", "唯一标识会话记录，不是用户 ID"),
        ("RevenueFlag", "整数", "Revenue=TRUE 为 1，否则为 0", "统计与建模目标变量"),
        ("MonthOrder", "整数", "自然月份序号 1-12", "Power BI 月份排序"),
        ("ProductDepthGroup", "文本", "0至5、6至10、11至20、21至49、50及以上", "产品页浏览深度分组"),
        ("ProductDepthOrder", "整数", "各浏览深度组对应 1-5", "浏览深度分组排序"),
        ("ProductDurationGroup", "文本", "按三分位划分为低、中、高", "产品页停留时长分组"),
        ("ProductDurationOrder", "整数", "低=1、中=2、高=3", "停留时长分组排序"),
        ("HighBounceSession", "布尔文本", "BounceRates 严格大于 P75", "标记高跳出会话"),
        ("HighExitSession", "布尔文本", "ExitRates 严格大于 P75", "标记高退出会话"),
    ]

    lines = [
        "# 电商网站会话分析派生字段说明",
        "",
        "## 1. 执行摘要",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 输入文件：`{input_path.name}`",
        f"- 输出文件：`{output_path.name}`",
        f"- 数据规模：{row_count:,} 行，{len(OUTPUT_COLUMNS)} 列",
        f"- 保留原始字段：{len(SOURCE_COLUMNS)} 个",
        f"- 新增派生字段：{len(DERIVED_COLUMNS)} 个",
        "- 处理结果：未删除记录，未修改任何原始字段值。",
        "",
        "## 2. 派生字段字典",
        "",
    ]
    lines.extend(markdown_table(["字段", "类型", "生成规则", "用途"], field_rows))

    lines.extend(
        [
            "",
            "## 3. 本次分位数阈值",
            "",
        ]
    )
    threshold_rows = [
        (
            "产品页停留时长 P33.33",
            fmt_number(thresholds["duration_p33"]),
            "小于或等于该值划为低",
        ),
        (
            "产品页停留时长 P66.67",
            fmt_number(thresholds["duration_p67"]),
            "大于 P33.33 且小于或等于该值划为中；其余为高",
        ),
        (
            "BounceRates P75",
            fmt_number(thresholds["bounce_p75"]),
            "严格大于该值标记为高跳出会话",
        ),
        (
            "ExitRates P75",
            fmt_number(thresholds["exit_p75"]),
            "严格大于该值标记为高退出会话",
        ),
    ]
    lines.extend(markdown_table(["阈值", "实际数值", "边界规则"], threshold_rows))
    lines.extend(
        [
            "",
            f"产品页停留时长为 0 的会话共有 {zero_duration_count:,} 条，占 {fmt_percent(zero_duration_count, row_count)}。两个三分位点互不相同，因此本次可以直接使用全体会话三分位数，不需要把零值单独成组。",
            "",
            "## 4. 产品页浏览深度分布",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["ProductDepthGroup", "会话数", "占比"],
            group_distribution(
                analysis_rows,
                "ProductDepthGroup",
                ["0至5", "6至10", "11至20", "21至49", "50及以上"],
            ),
        )
    )

    lines.extend(["", "## 5. 产品页停留时长分布", ""])
    lines.extend(
        markdown_table(
            ["ProductDurationGroup", "会话数", "占比"],
            group_distribution(
                analysis_rows, "ProductDurationGroup", ["低", "中", "高"]
            ),
        )
    )

    lines.extend(["", "## 6. 高跳出与高退出标记分布", ""])
    high_flag_rows = []
    for column in ["HighBounceSession", "HighExitSession"]:
        counts = Counter(str(row[column]) for row in analysis_rows)
        high_flag_rows.append(
            (column, f"{counts['TRUE']:,}", fmt_percent(counts["TRUE"], row_count))
        )
    lines.extend(markdown_table(["字段", "TRUE 会话数", "TRUE 占比"], high_flag_rows))
    lines.extend(
        [
            "",
            "由于使用“严格大于 P75”而不是“大于或等于”，且边界处可能有重复值，实际标记比例不要求恰好等于 25%。",
            "",
            "## 7. 月份排序映射",
            "",
        ]
    )
    observed_months = sorted(
        {row["Month"] for row in source_rows}, key=lambda value: MONTH_ORDER[value]
    )
    lines.extend(
        markdown_table(
            ["Month", "MonthOrder"],
            [(month, MONTH_ORDER[month]) for month in observed_months],
        )
    )

    lines.extend(
        [
            "",
            "## 8. 数据验证",
            "",
            f"- `SessionID` 从 1 连续递增至 {row_count:,}，不存在重复或缺口。",
            "- `RevenueFlag` 已逐行与 `Revenue` 核对。",
            "- `MonthOrder` 已逐行与月份映射核对。",
            "- 所有原始字段和值均与输入数据一致。",
            "- 输出行数与输入行数一致。",
            f"- 输入文件 SHA-256：`{sha256(input_path)}`",
            f"- 输出文件 SHA-256：`{sha256(output_path)}`",
            "",
            "## 9. Power BI 使用要求",
            "",
            "- 将 `Month` 设置为按 `MonthOrder` 排序。",
            "- 将 `ProductDepthGroup` 设置为按 `ProductDepthOrder` 排序。",
            "- 将 `ProductDurationGroup` 设置为按 `ProductDurationOrder` 排序。",
            "- `RevenueFlag` 作为 0/1 数值字段；`Revenue`、`Weekend`、`HighBounceSession` 和 `HighExitSession` 作为布尔字段使用。",
            "- `SessionID` 只用于会话计数和明细定位，不得作为用户 ID 使用。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    source_rows = read_source(input_path)
    analysis_rows, thresholds = build_analysis_rows(source_rows)
    validate_output(source_rows, analysis_rows)
    write_output(output_path, analysis_rows)

    report = build_report(
        input_path, output_path, source_rows, analysis_rows, thresholds
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows: {len(analysis_rows):,}")
    print(f"Source columns: {len(SOURCE_COLUMNS)}")
    print(f"Derived columns: {len(DERIVED_COLUMNS)}")
    print(f"Output columns: {len(OUTPUT_COLUMNS)}")
    print(f"Output CSV: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
