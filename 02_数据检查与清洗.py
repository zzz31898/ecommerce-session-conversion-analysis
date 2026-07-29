"""Audit and minimally clean the online shoppers session dataset.

The raw file is never modified. Exact duplicate rows and statistical outliers
are reported but retained because the source has no natural session key.
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


EXPECTED_COLUMNS = [
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

INTEGER_COLUMNS = {
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
}

NUMERIC_COLUMNS = [
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
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

BEHAVIOR_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]

RATE_COLUMNS = {"BounceRates", "ExitRates", "SpecialDay"}
BOOLEAN_COLUMNS = {"Weekend", "Revenue"}
VALID_BOOLEAN_VALUES = {"TRUE", "FALSE"}
VALID_VISITOR_TYPES = {"New_Visitor", "Returning_Visitor", "Other"}
VALID_MONTHS = {
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "online_shoppers_intention.csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "processed" / "online_shoppers_cleaned_base.csv",
        help="Path for the minimally cleaned base CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "02_数据质量报告.md",
        help="Path for the Markdown quality report.",
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
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fmt_number(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def fmt_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    return f"{numerator / denominator:.2%}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def read_and_normalize(input_path: Path) -> tuple[list[dict[str, str]], int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected CSV columns.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Actual:   {reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []
        trimmed_cells = 0
        for source_row in reader:
            clean_row: dict[str, str] = {}
            for column in EXPECTED_COLUMNS:
                source_value = source_row[column]
                clean_value = source_value.strip() if source_value is not None else ""
                trimmed_cells += int(clean_value != source_value)
                clean_row[column] = clean_value
            rows.append(clean_row)
    return rows, trimmed_cells


def audit_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    missing_counts = {
        column: sum(row[column] == "" for row in rows) for column in EXPECTED_COLUMNS
    }

    invalid_numeric: dict[str, int] = {column: 0 for column in NUMERIC_COLUMNS}
    non_integer: dict[str, int] = {column: 0 for column in INTEGER_COLUMNS}
    negative_counts: dict[str, int] = {column: 0 for column in NUMERIC_COLUMNS}
    range_violations: dict[str, int] = {column: 0 for column in RATE_COLUMNS}
    numeric_values: dict[str, list[float]] = {column: [] for column in NUMERIC_COLUMNS}

    for row in rows:
        for column in NUMERIC_COLUMNS:
            raw_value = row[column]
            if raw_value == "":
                continue
            try:
                value = float(raw_value)
            except ValueError:
                invalid_numeric[column] += 1
                continue
            if not math.isfinite(value):
                invalid_numeric[column] += 1
                continue
            numeric_values[column].append(value)
            negative_counts[column] += int(value < 0)
            if column in INTEGER_COLUMNS:
                non_integer[column] += int(not value.is_integer())
            if column in RATE_COLUMNS:
                range_violations[column] += int(value < 0 or value > 1)

    invalid_boolean = {
        column: sum(
            row[column] != "" and row[column] not in VALID_BOOLEAN_VALUES for row in rows
        )
        for column in BOOLEAN_COLUMNS
    }
    invalid_visitor_type = sum(
        row["VisitorType"] != "" and row["VisitorType"] not in VALID_VISITOR_TYPES
        for row in rows
    )
    invalid_month = sum(
        row["Month"] != "" and row["Month"] not in VALID_MONTHS for row in rows
    )

    row_counter = Counter(
        tuple(row[column] for column in EXPECTED_COLUMNS) for row in rows
    )
    duplicate_groups = sum(count > 1 for count in row_counter.values())
    duplicate_extra_rows = sum(count - 1 for count in row_counter.values() if count > 1)
    maximum_duplicate_copies = max(row_counter.values(), default=0)

    category_counts = {
        column: Counter(row[column] for row in rows)
        for column in ["Revenue", "Weekend", "VisitorType", "Month"]
    }

    return {
        "missing_counts": missing_counts,
        "invalid_numeric": invalid_numeric,
        "non_integer": non_integer,
        "negative_counts": negative_counts,
        "range_violations": range_violations,
        "invalid_boolean": invalid_boolean,
        "invalid_visitor_type": invalid_visitor_type,
        "invalid_month": invalid_month,
        "numeric_values": numeric_values,
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "maximum_duplicate_copies": maximum_duplicate_copies,
        "category_counts": category_counts,
    }


def write_clean_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    trimmed_cells: int,
    audit: dict[str, object],
) -> str:
    row_count = len(rows)
    missing_counts = audit["missing_counts"]
    invalid_numeric = audit["invalid_numeric"]
    non_integer = audit["non_integer"]
    negative_counts = audit["negative_counts"]
    range_violations = audit["range_violations"]
    invalid_boolean = audit["invalid_boolean"]
    numeric_values = audit["numeric_values"]
    category_counts = audit["category_counts"]

    total_missing = sum(missing_counts.values())
    total_invalid_numeric = sum(invalid_numeric.values())
    total_non_integer = sum(non_integer.values())
    total_negative = sum(negative_counts.values())
    total_range_violations = sum(range_violations.values())
    total_invalid_boolean = sum(invalid_boolean.values())
    total_invalid_categories = audit["invalid_visitor_type"] + audit["invalid_month"]

    lines = [
        "# 电商网站会话数据质量报告",
        "",
        "## 1. 执行摘要",
        "",
        f"- 检查时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 原始文件：`{input_path.name}`",
        f"- 数据规模：{row_count:,} 行，{len(EXPECTED_COLUMNS)} 列",
        f"- 清洗后文件：`{output_path.name}`",
        "- 处理原则：保留原始文件；只标准化字段前后空白；不自动删除重复行或极端值。",
        "",
    ]

    issue_rows = [
        ("缺失单元格", f"{total_missing:,}", "无缺失则不填补"),
        ("无法解析的数值", f"{total_invalid_numeric:,}", "发现时停止下游分析并人工核查"),
        ("整数列中的非整数值", f"{total_non_integer:,}", "发现时人工核查"),
        ("负数值", f"{total_negative:,}", "当前字段应为非负，发现时人工核查"),
        ("比例字段越界", f"{total_range_violations:,}", "要求位于 0 至 1"),
        ("无效布尔值", f"{total_invalid_boolean:,}", "只接受 TRUE/FALSE"),
        ("无效月份或访客类型", f"{total_invalid_categories:,}", "不擅自归类"),
        ("完全重复的额外记录", f"{audit['duplicate_extra_rows']:,}", "保留并单独说明"),
        ("被标准化空白的单元格", f"{trimmed_cells:,}", "去除字段值前后空白"),
    ]
    lines.extend(markdown_table(["检查项", "数量", "处理决定"], issue_rows))

    lines.extend(
        [
            "",
            "## 2. 缺失值检查",
            "",
        ]
    )
    missing_rows = [
        (column, f"{missing_counts[column]:,}", fmt_percent(missing_counts[column], row_count))
        for column in EXPECTED_COLUMNS
    ]
    lines.extend(markdown_table(["字段", "缺失数", "缺失率"], missing_rows))

    lines.extend(
        [
            "",
            "## 3. 重复记录检查",
            "",
            f"- 完全重复记录组数：{audit['duplicate_groups']:,}。",
            f"- 完全重复的额外记录数：{audit['duplicate_extra_rows']:,}。",
            f"- 单一相同记录的最大出现次数：{audit['maximum_duplicate_copies']:,}。",
            "- 处理决定：全部保留。数据没有原始会话编号或用户 ID，完全相同的行为记录仍可能来自不同会话；直接删除会造成无法验证的样本损失。",
            "- 后续动作：新增 `SessionID` 后把这些记录视为不同会话，同时在分析说明中保留这一数据限制。",
            "",
            "## 4. 数值范围与零值",
            "",
        ]
    )

    numeric_rows = []
    for column in NUMERIC_COLUMNS:
        values = numeric_values[column]
        numeric_rows.append(
            (
                column,
                fmt_number(min(values)) if values else "NA",
                fmt_number(quantile(values, 0.5)),
                fmt_number(quantile(values, 0.99)),
                fmt_number(max(values)) if values else "NA",
                fmt_percent(sum(value == 0 for value in values), len(values)),
            )
        )
    lines.extend(
        markdown_table(
            ["字段", "最小值", "中位数", "P99", "最大值", "零值占比"],
            numeric_rows,
        )
    )

    lines.extend(
        [
            "",
            "比例字段 `BounceRates`、`ExitRates`、`SpecialDay` 的合法范围定义为 0 至 1；所有页面数量、停留时间、编码和页面价值字段按非负值检查。",
            "",
            "## 5. 异常分布检查",
            "",
        ]
    )
    outlier_rows = []
    for column in BEHAVIOR_COLUMNS:
        values = numeric_values[column]
        q1 = quantile(values, 0.25)
        median = quantile(values, 0.5)
        q3 = quantile(values, 0.75)
        upper_fence = q3 + 1.5 * (q3 - q1)
        outlier_count = sum(value > upper_fence for value in values)
        outlier_rows.append(
            (
                column,
                fmt_number(q1),
                fmt_number(median),
                fmt_number(q3),
                fmt_number(upper_fence),
                f"{outlier_count:,}",
                fmt_percent(outlier_count, len(values)),
            )
        )
    lines.extend(
        markdown_table(
            ["字段", "Q1", "中位数", "Q3", "IQR 上界", "上界外数量", "占比"],
            outlier_rows,
        )
    )
    lines.extend(
        [
            "",
            "IQR 上界外记录仅用于描述偏态和长尾，不等同于错误数据。页面浏览量、停留时间、跳出率、退出率和 `PageValues` 均可能自然偏态，本阶段不截尾、不删除；后续统计检验优先使用稳健方法，建模时再依据模型需要决定是否变换。",
            "",
            "## 6. 类别分布",
            "",
        ]
    )

    for column in ["Revenue", "Weekend", "VisitorType", "Month"]:
        lines.extend([f"### 6.{['Revenue', 'Weekend', 'VisitorType', 'Month'].index(column) + 1} `{column}`", ""])
        counts = category_counts[column]
        category_rows = [
            (value or "(空值)", f"{count:,}", fmt_percent(count, row_count))
            for value, count in sorted(counts.items())
        ]
        lines.extend(markdown_table(["取值", "会话数", "占比"], category_rows))
        lines.append("")

    revenue_counts = category_counts["Revenue"]
    conversion_rate = revenue_counts.get("TRUE", 0) / row_count if row_count else math.nan
    lines.extend(
        [
            "## 7. 类别不平衡提示",
            "",
            f"购买会话为 {revenue_counts.get('TRUE', 0):,} 条，未购买会话为 {revenue_counts.get('FALSE', 0):,} 条，当前样本会话购买转化率为 {conversion_rate:.2%}。",
            "",
            "购买类别明显少于未购买类别。后续预测模型不能只看准确率，应至少报告 Precision、Recall、F1、ROC-AUC、PR-AUC 和混淆矩阵，并使用分层划分保留类别比例。",
            "",
            "## 8. 本阶段实际清洗动作",
            "",
            f"1. 校验并保留固定的 {len(EXPECTED_COLUMNS)} 个原始字段。",
            f"2. 去除字段值前后空白，共处理 {trimmed_cells:,} 个单元格。",
            "3. 保持 `Revenue` 和 `Weekend` 为统一的 TRUE/FALSE 文本表示，方便 CSV 与 Power BI 读取。",
            "4. 未填补缺失值，未删除任何记录，未处理统计极端值。",
            "5. 未在本阶段生成 `SessionID`、行为分组或月份排序字段，这些属于下一阶段的字段工程。",
            "",
            "## 9. 文件校验",
            "",
            f"- 原始文件 SHA-256：`{sha256(input_path)}`",
            f"- 清洗后文件 SHA-256：`{sha256(output_path)}`",
            f"- 清洗后数据规模：{row_count:,} 行，{len(EXPECTED_COLUMNS)} 列。",
            "- 原始文件未被修改。",
            "",
            "## 10. 第二阶段结论",
            "",
            "当前数据可以进入字段工程阶段。完全重复记录和行为字段长尾是主要需要保留说明的数据特征，不应在缺少会话主键和业务证据时直接删除。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_before_output(audit: dict[str, object]) -> None:
    blocking_issues = {
        "invalid_numeric": sum(audit["invalid_numeric"].values()),
        "non_integer": sum(audit["non_integer"].values()),
        "negative_values": sum(audit["negative_counts"].values()),
        "rate_range_violations": sum(audit["range_violations"].values()),
        "invalid_boolean": sum(audit["invalid_boolean"].values()),
        "invalid_visitor_type": audit["invalid_visitor_type"],
        "invalid_month": audit["invalid_month"],
    }
    present_issues = {name: count for name, count in blocking_issues.items() if count}
    if present_issues:
        raise ValueError(f"Blocking data quality issues found: {present_issues}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    rows, trimmed_cells = read_and_normalize(input_path)
    if not rows:
        raise ValueError("The input CSV contains no data rows.")

    audit = audit_rows(rows)
    validate_before_output(audit)
    write_clean_csv(output_path, rows)
    report = build_report(input_path, output_path, rows, trimmed_cells, audit)
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows: {len(rows):,}")
    print(f"Columns: {len(EXPECTED_COLUMNS)}")
    print(f"Duplicate extra rows retained: {audit['duplicate_extra_rows']:,}")
    print(f"Clean CSV: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
