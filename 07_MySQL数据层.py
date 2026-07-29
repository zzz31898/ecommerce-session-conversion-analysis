from __future__ import annotations

import argparse
import csv
import getpass
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional, Sequence

try:
    import mysql.connector as mysql_connector
except ModuleNotFoundError:
    mysql_connector = None


PROJECT_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = PROJECT_DIR / "sql" / "07_mysql_schema.sql"
VIEWS_FILE = PROJECT_DIR / "sql" / "07_mysql_views.sql"
DEFAULT_DATABASE = "online_shoppers_analytics"
STATEMENT_SEPARATOR = re.compile(r"^\s*-- statement-break\s*$", re.MULTILINE)


class DataValidationError(ValueError):
    pass


def as_text(value: str) -> str:
    return value.strip()


def as_int(value: str) -> int:
    return int(value)


def as_float(value: str) -> float:
    return float(value)


def as_decimal(value: str) -> Decimal:
    return Decimal(value)


def as_optional_decimal(value: str) -> Optional[Decimal]:
    value = value.strip()
    return None if value == "" else Decimal(value)


def as_bool_int(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"unsupported Boolean value: {value!r}")


@dataclass(frozen=True)
class ColumnSpec:
    target: str
    source: str
    converter: Callable[[str], object]


@dataclass(frozen=True)
class LoadSpec:
    table: str
    csv_path: Path
    columns: tuple[ColumnSpec, ...]


@dataclass
class LoadedTable:
    spec: LoadSpec
    rows: list[tuple[object, ...]]

    @property
    def target_columns(self) -> list[str]:
        return [column.target for column in self.spec.columns]

    def column_index(self, name: str) -> int:
        return self.target_columns.index(name)


def column(name: str, converter: Callable[[str], object]) -> ColumnSpec:
    return ColumnSpec(name, name, converter)


FACT_SESSION_COLUMNS = (
    column("SessionID", as_int),
    column("Administrative", as_int),
    column("Administrative_Duration", as_float),
    column("Informational", as_int),
    column("Informational_Duration", as_float),
    column("ProductRelated", as_int),
    column("ProductRelated_Duration", as_float),
    column("BounceRates", as_float),
    column("ExitRates", as_float),
    column("PageValues", as_float),
    column("SpecialDay", as_float),
    column("Month", as_text),
    column("OperatingSystems", as_int),
    column("Browser", as_int),
    column("Region", as_int),
    column("TrafficType", as_int),
    column("VisitorType", as_text),
    column("Weekend", as_bool_int),
    column("Revenue", as_bool_int),
    column("RevenueFlag", as_int),
    column("MonthOrder", as_int),
    column("ProductDepthGroup", as_text),
    column("ProductDepthOrder", as_int),
    column("ProductDurationGroup", as_text),
    column("ProductDurationOrder", as_int),
    column("HighBounceSession", as_bool_int),
    column("HighExitSession", as_bool_int),
)

MODEL_SCORE_COLUMNS = (
    column("SessionID", as_int),
    column("RevenueFlag", as_int),
    column("BehaviorExclPageValuesOutOfSampleScore", as_decimal),
    column("BehaviorExclPageValuesScoreSource", as_text),
    column("PageValuesBenchmarkOutOfSampleScore", as_decimal),
    column("PageValuesBenchmarkScoreSource", as_text),
    column(
        "BehaviorExclPageValuesPercentileAmongUnpurchased",
        as_optional_decimal,
    ),
    column("HistoricalHighPotentialTop10Percent", as_bool_int),
)

MODEL_EVALUATION_COLUMNS = (
    column("Scenario", as_text),
    column("ScenarioName", as_text),
    column("Model", as_text),
    column("ModelName", as_text),
    column("Threshold", as_decimal),
    column("Accuracy", as_decimal),
    ColumnSpec("PrecisionValue", "Precision", as_decimal),
    ColumnSpec("RecallValue", "Recall", as_decimal),
    column("F1", as_decimal),
    column("ROCAUC", as_decimal),
    column("PRAUC", as_decimal),
    column("BrierScore", as_decimal),
)

OPERATING_POINT_COLUMNS = (
    column("Scenario", as_text),
    column("Model", as_text),
    column("SelectedTopPercent", as_decimal),
    column("SelectedSessions", as_int),
    column("ScoreThresholdAtCoverage", as_decimal),
    column("PurchasesCaptured", as_int),
    ColumnSpec("PrecisionValue", "Precision", as_decimal),
    ColumnSpec("RecallValue", "Recall", as_decimal),
    column("LiftVsTestBaseline", as_decimal),
)

DECILE_LIFT_COLUMNS = (
    column("Scenario", as_text),
    column("Model", as_text),
    column("ScoreDecileHighToLow", as_int),
    column("Sessions", as_int),
    column("Purchases", as_int),
    column("MeanScore", as_decimal),
    column("ObservedConversionRate", as_decimal),
    column("LiftVsTestBaseline", as_decimal),
    column("CumulativeSessions", as_int),
    column("CumulativePurchases", as_int),
    column("CumulativeRecall", as_decimal),
)

CALIBRATION_COLUMNS = (
    column("Scenario", as_text),
    column("Model", as_text),
    column("ScoreBinLowToHigh", as_int),
    column("Sessions", as_int),
    column("Purchases", as_int),
    column("MeanScore", as_decimal),
    column("ObservedConversionRate", as_decimal),
    column("CalibrationGapObservedMinusScore", as_decimal),
)

METRIC_CI_COLUMNS = (
    column("Scenario", as_text),
    column("ScenarioName", as_text),
    column("Metric", as_text),
    column("Lower95", as_decimal),
    column("Upper95", as_decimal),
    column("BootstrapIterations", as_int),
)

FEATURE_AVAILABILITY_COLUMNS = (
    column("FeatureGroup", as_text),
    column("Fields", as_text),
    column("Availability", as_text),
    column("AllowedForRealTimeEarlyIntervention", as_text),
    column("Decision", as_text),
)

LOAD_SPECS = (
    LoadSpec(
        "fact_sessions",
        PROJECT_DIR / "online_shoppers_analysis_ready_v2.csv",
        FACT_SESSION_COLUMNS,
    ),
    LoadSpec(
        "fact_model_scores",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "out_of_sample_session_scores.csv",
        MODEL_SCORE_COLUMNS,
    ),
    LoadSpec(
        "model_evaluation",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "champion_diagnostic_metrics.csv",
        MODEL_EVALUATION_COLUMNS,
    ),
    LoadSpec(
        "model_operating_points",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "operating_points.csv",
        OPERATING_POINT_COLUMNS,
    ),
    LoadSpec(
        "model_decile_lift",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "test_decile_lift.csv",
        DECILE_LIFT_COLUMNS,
    ),
    LoadSpec(
        "model_calibration_bins",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "calibration_bins.csv",
        CALIBRATION_COLUMNS,
    ),
    LoadSpec(
        "model_metric_ci",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "bootstrap_confidence_intervals.csv",
        METRIC_CI_COLUMNS,
    ),
    LoadSpec(
        "model_feature_availability",
        PROJECT_DIR
        / "outputs"
        / "models"
        / "enhanced_evaluation"
        / "feature_availability.csv",
        FEATURE_AVAILABILITY_COLUMNS,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and load the MySQL analytics data layer."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument(
        "--password-env",
        default="MYSQL_PASSWORD",
        help="Environment variable holding the password. Prompts when absent.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Create tables and views without replacing table data.",
    )
    return parser.parse_args()


def validate_database_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(
            "Database name may contain only ASCII letters, numbers, and underscores."
        )


def load_csv(spec: LoadSpec) -> LoadedTable:
    if not spec.csv_path.exists():
        raise DataValidationError(f"Missing source file: {spec.csv_path}")

    expected_headers = [column.source for column in spec.columns]
    rows: list[tuple[object, ...]] = []
    with spec.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_headers:
            raise DataValidationError(
                f"Header mismatch in {spec.csv_path}.\n"
                f"Expected: {expected_headers}\nActual:   {reader.fieldnames}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            try:
                rows.append(
                    tuple(
                        column.converter(raw_row[column.source])
                        for column in spec.columns
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"Invalid value in {spec.csv_path}, line {line_number}: {exc}"
                ) from exc

    if not rows:
        raise DataValidationError(f"Source file contains no data rows: {spec.csv_path}")
    return LoadedTable(spec=spec, rows=rows)


def validate_loaded_data(tables: Sequence[LoadedTable]) -> None:
    by_name = {table.spec.table: table for table in tables}
    sessions = by_name["fact_sessions"]
    scores = by_name["fact_model_scores"]

    session_id_index = sessions.column_index("SessionID")
    session_revenue_index = sessions.column_index("Revenue")
    session_flag_index = sessions.column_index("RevenueFlag")
    session_ids = [int(row[session_id_index]) for row in sessions.rows]

    if len(session_ids) != len(set(session_ids)):
        raise DataValidationError("fact_sessions contains duplicate SessionID values.")
    if session_ids != list(range(1, len(session_ids) + 1)):
        raise DataValidationError("SessionID must be continuous and start at 1.")
    if any(
        int(row[session_revenue_index]) != int(row[session_flag_index])
        for row in sessions.rows
    ):
        raise DataValidationError("Revenue and RevenueFlag are inconsistent.")

    score_id_index = scores.column_index("SessionID")
    score_flag_index = scores.column_index("RevenueFlag")
    high_potential_index = scores.column_index(
        "HistoricalHighPotentialTop10Percent"
    )
    score_ids = [int(row[score_id_index]) for row in scores.rows]
    if len(score_ids) != len(set(score_ids)):
        raise DataValidationError("fact_model_scores contains duplicate SessionID values.")
    if set(score_ids) != set(session_ids):
        raise DataValidationError(
            "SessionID coverage differs between session data and model scores."
        )

    revenue_by_session = {
        int(row[session_id_index]): int(row[session_flag_index])
        for row in sessions.rows
    }
    for row in scores.rows:
        session_id = int(row[score_id_index])
        score_revenue = int(row[score_flag_index])
        high_potential = int(row[high_potential_index])
        if score_revenue != revenue_by_session[session_id]:
            raise DataValidationError(
                f"RevenueFlag mismatch in model scores for SessionID={session_id}."
            )
        if high_potential == 1 and score_revenue != 0:
            raise DataValidationError(
                "Historical high-potential flag must only identify unpurchased sessions."
            )


def read_sql_statements(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8-sig")
    statements = [
        statement.strip()
        for statement in STATEMENT_SEPARATOR.split(content)
        if statement.strip()
    ]
    if not statements:
        raise ValueError(f"No SQL statements found in {path}")
    return statements


def execute_sql_file(connection: object, path: Path) -> None:
    cursor = connection.cursor()
    try:
        for statement in read_sql_statements(path):
            cursor.execute(statement)
    finally:
        cursor.close()


def insert_rows(connection: object, table: LoadedTable, chunk_size: int = 1000) -> None:
    columns = table.target_columns
    placeholders = ", ".join(["%s"] * len(columns))
    quoted_columns = ", ".join(f"`{name}`" for name in columns)
    sql = f"INSERT INTO `{table.spec.table}` ({quoted_columns}) VALUES ({placeholders})"

    cursor = connection.cursor()
    try:
        for start in range(0, len(table.rows), chunk_size):
            cursor.executemany(sql, table.rows[start : start + chunk_size])
    finally:
        cursor.close()


def replace_table_data(connection: object, tables: Sequence[LoadedTable]) -> None:
    delete_order = (
        "fact_model_scores",
        "fact_sessions",
        "model_evaluation",
        "model_operating_points",
        "model_decile_lift",
        "model_calibration_bins",
        "model_metric_ci",
        "model_feature_availability",
    )
    cursor = connection.cursor()
    try:
        for table_name in delete_order:
            cursor.execute(f"DELETE FROM `{table_name}`")
    finally:
        cursor.close()

    for table in tables:
        insert_rows(connection, table)
        print(f"Loaded {table.spec.table}: {len(table.rows):,} rows")


def verify_database(connection: object, expected: dict[str, int]) -> None:
    cursor = connection.cursor()
    try:
        for table_name, expected_count in expected.items():
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            actual_count = int(cursor.fetchone()[0])
            if actual_count != expected_count:
                raise DataValidationError(
                    f"Row-count mismatch for {table_name}: "
                    f"expected {expected_count}, got {actual_count}."
                )

        cursor.execute(
            """
            SELECT
                COUNT(*),
                SUM(RevenueFlag),
                AVG(RevenueFlag),
                SUM(Revenue <> RevenueFlag)
            FROM fact_sessions
            """
        )
        session_count, purchases, conversion_rate, flag_mismatches = cursor.fetchone()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM fact_sessions AS s
            LEFT JOIN fact_model_scores AS ms ON s.SessionID = ms.SessionID
            WHERE ms.SessionID IS NULL
            """
        )
        missing_scores = int(cursor.fetchone()[0])
        if int(flag_mismatches) != 0 or missing_scores != 0:
            raise DataValidationError(
                "Database verification found target or model-score integrity errors."
            )

        print(
            "Verified fact_sessions: "
            f"sessions={int(session_count):,}, purchases={int(purchases):,}, "
            f"conversion_rate={float(conversion_rate):.4%}"
        )
    finally:
        cursor.close()


def connect_server(args: argparse.Namespace, password: str) -> object:
    return mysql_connector.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password,
        connection_timeout=10,
        charset="utf8mb4",
        use_unicode=True,
    )


def create_database(args: argparse.Namespace, password: str) -> None:
    connection = connect_server(args, password)
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{args.database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    finally:
        cursor.close()
        connection.close()


def connect_database(args: argparse.Namespace, password: str) -> object:
    connection = connect_server(args, password)
    connection.database = args.database
    return connection


def run() -> int:
    args = parse_args()
    validate_database_name(args.database)

    if mysql_connector is None:
        print(
            "Missing mysql-connector-python. Install it with:\n"
            "  python -m pip install -r requirements-sql.txt",
            file=sys.stderr,
        )
        return 2

    loaded_tables: list[LoadedTable] = []
    if not args.schema_only:
        print("Validating source CSV files...")
        loaded_tables = [load_csv(spec) for spec in LOAD_SPECS]
        validate_loaded_data(loaded_tables)
        print("Source validation passed.")

    password = os.environ.get(args.password_env)
    if password is None:
        password = getpass.getpass(
            f"MySQL password for {args.user}@{args.host}:{args.port}: "
        )

    create_database(args, password)
    connection = connect_database(args, password)
    try:
        execute_sql_file(connection, SCHEMA_FILE)
        connection.commit()
        print(f"Schema ready in database: {args.database}")

        if loaded_tables:
            try:
                connection.start_transaction()
                replace_table_data(connection, loaded_tables)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        execute_sql_file(connection, VIEWS_FILE)
        connection.commit()
        print("Power BI views ready.")

        if loaded_tables:
            verify_database(
                connection,
                {table.spec.table: len(table.rows) for table in loaded_tables},
            )
    finally:
        connection.close()

    print("MySQL analytics data layer completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (DataValidationError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"MySQL load failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
