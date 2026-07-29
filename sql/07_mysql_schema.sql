-- MySQL 8.0 schema for the session behavior and purchase conversion project.
-- Select the target database before running this file manually in Navicat.
-- The Python loader creates and selects the database automatically.

CREATE TABLE IF NOT EXISTS dim_month (
    MonthOrder TINYINT UNSIGNED NOT NULL,
    MonthCode VARCHAR(4) NOT NULL,
    MonthNameChinese VARCHAR(4) NOT NULL,
    PRIMARY KEY (MonthOrder),
    UNIQUE KEY uq_dim_month_code (MonthCode),
    CONSTRAINT chk_dim_month_order CHECK (MonthOrder BETWEEN 1 AND 12)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

INSERT INTO dim_month (MonthOrder, MonthCode, MonthNameChinese) VALUES
    (1, 'Jan', '一月'),
    (2, 'Feb', '二月'),
    (3, 'Mar', '三月'),
    (4, 'Apr', '四月'),
    (5, 'May', '五月'),
    (6, 'June', '六月'),
    (7, 'Jul', '七月'),
    (8, 'Aug', '八月'),
    (9, 'Sep', '九月'),
    (10, 'Oct', '十月'),
    (11, 'Nov', '十一月'),
    (12, 'Dec', '十二月')
ON DUPLICATE KEY UPDATE
    MonthCode = VALUES(MonthCode),
    MonthNameChinese = VALUES(MonthNameChinese);

-- statement-break

CREATE TABLE IF NOT EXISTS fact_sessions (
    SessionID INT UNSIGNED NOT NULL,
    Administrative SMALLINT UNSIGNED NOT NULL,
    Administrative_Duration DOUBLE NOT NULL,
    Informational SMALLINT UNSIGNED NOT NULL,
    Informational_Duration DOUBLE NOT NULL,
    ProductRelated SMALLINT UNSIGNED NOT NULL,
    ProductRelated_Duration DOUBLE NOT NULL,
    BounceRates DOUBLE NOT NULL,
    ExitRates DOUBLE NOT NULL,
    PageValues DOUBLE NOT NULL,
    SpecialDay DOUBLE NOT NULL,
    Month VARCHAR(4) NOT NULL,
    OperatingSystems SMALLINT UNSIGNED NOT NULL,
    Browser SMALLINT UNSIGNED NOT NULL,
    Region SMALLINT UNSIGNED NOT NULL,
    TrafficType SMALLINT UNSIGNED NOT NULL,
    VisitorType VARCHAR(32) NOT NULL,
    Weekend TINYINT(1) NOT NULL,
    Revenue TINYINT(1) NOT NULL,
    RevenueFlag TINYINT UNSIGNED NOT NULL,
    MonthOrder TINYINT UNSIGNED NOT NULL,
    ProductDepthGroup VARCHAR(16) NOT NULL,
    ProductDepthOrder TINYINT UNSIGNED NOT NULL,
    ProductDurationGroup VARCHAR(8) NOT NULL,
    ProductDurationOrder TINYINT UNSIGNED NOT NULL,
    HighBounceSession TINYINT(1) NOT NULL,
    HighExitSession TINYINT(1) NOT NULL,
    LoadedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (SessionID),
    KEY idx_sessions_revenue (RevenueFlag),
    KEY idx_sessions_month (MonthOrder, Month),
    KEY idx_sessions_visitor (VisitorType),
    KEY idx_sessions_traffic (TrafficType),
    KEY idx_sessions_weekend (Weekend),
    KEY idx_sessions_product_depth (ProductDepthOrder),
    KEY idx_sessions_product_duration (ProductDurationOrder),
    KEY idx_sessions_bounce_exit (HighBounceSession, HighExitSession),
    CONSTRAINT fk_sessions_month
        FOREIGN KEY (Month) REFERENCES dim_month (MonthCode),
    CONSTRAINT chk_sessions_revenue CHECK (Revenue IN (0, 1)),
    CONSTRAINT chk_sessions_revenue_flag CHECK (RevenueFlag IN (0, 1)),
    CONSTRAINT chk_sessions_weekend CHECK (Weekend IN (0, 1)),
    CONSTRAINT chk_sessions_high_bounce CHECK (HighBounceSession IN (0, 1)),
    CONSTRAINT chk_sessions_high_exit CHECK (HighExitSession IN (0, 1)),
    CONSTRAINT chk_sessions_month_order CHECK (MonthOrder BETWEEN 1 AND 12),
    CONSTRAINT chk_sessions_product_depth_order CHECK (ProductDepthOrder BETWEEN 1 AND 5),
    CONSTRAINT chk_sessions_product_duration_order CHECK (ProductDurationOrder BETWEEN 1 AND 3)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS fact_model_scores (
    SessionID INT UNSIGNED NOT NULL,
    RevenueFlag TINYINT UNSIGNED NOT NULL,
    BehaviorExclPageValuesOutOfSampleScore DECIMAL(18, 12) NOT NULL,
    BehaviorExclPageValuesScoreSource VARCHAR(24) NOT NULL,
    PageValuesBenchmarkOutOfSampleScore DECIMAL(18, 12) NOT NULL,
    PageValuesBenchmarkScoreSource VARCHAR(24) NOT NULL,
    BehaviorExclPageValuesPercentileAmongUnpurchased DECIMAL(18, 12) NULL,
    HistoricalHighPotentialTop10Percent TINYINT(1) NOT NULL,
    LoadedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (SessionID),
    KEY idx_scores_high_potential (HistoricalHighPotentialTop10Percent),
    KEY idx_scores_behavior_score (BehaviorExclPageValuesOutOfSampleScore),
    CONSTRAINT fk_scores_session
        FOREIGN KEY (SessionID) REFERENCES fact_sessions (SessionID)
        ON DELETE CASCADE,
    CONSTRAINT chk_scores_revenue_flag CHECK (RevenueFlag IN (0, 1)),
    CONSTRAINT chk_scores_high_potential CHECK (HistoricalHighPotentialTop10Percent IN (0, 1))
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_evaluation (
    Scenario VARCHAR(40) NOT NULL,
    ScenarioName VARCHAR(80) NOT NULL,
    Model VARCHAR(40) NOT NULL,
    ModelName VARCHAR(40) NOT NULL,
    Threshold DECIMAL(18, 12) NOT NULL,
    Accuracy DECIMAL(18, 12) NOT NULL,
    PrecisionValue DECIMAL(18, 12) NOT NULL,
    RecallValue DECIMAL(18, 12) NOT NULL,
    F1 DECIMAL(18, 12) NOT NULL,
    ROCAUC DECIMAL(18, 12) NOT NULL,
    PRAUC DECIMAL(18, 12) NOT NULL,
    BrierScore DECIMAL(18, 12) NOT NULL,
    PRIMARY KEY (Scenario, Model)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_operating_points (
    Scenario VARCHAR(40) NOT NULL,
    Model VARCHAR(40) NOT NULL,
    SelectedTopPercent DECIMAL(8, 6) NOT NULL,
    SelectedSessions INT UNSIGNED NOT NULL,
    ScoreThresholdAtCoverage DECIMAL(18, 12) NOT NULL,
    PurchasesCaptured INT UNSIGNED NOT NULL,
    PrecisionValue DECIMAL(18, 12) NOT NULL,
    RecallValue DECIMAL(18, 12) NOT NULL,
    LiftVsTestBaseline DECIMAL(18, 12) NOT NULL,
    PRIMARY KEY (Scenario, Model, SelectedTopPercent)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_decile_lift (
    Scenario VARCHAR(40) NOT NULL,
    Model VARCHAR(40) NOT NULL,
    ScoreDecileHighToLow TINYINT UNSIGNED NOT NULL,
    Sessions INT UNSIGNED NOT NULL,
    Purchases INT UNSIGNED NOT NULL,
    MeanScore DECIMAL(18, 12) NOT NULL,
    ObservedConversionRate DECIMAL(18, 12) NOT NULL,
    LiftVsTestBaseline DECIMAL(18, 12) NOT NULL,
    CumulativeSessions INT UNSIGNED NOT NULL,
    CumulativePurchases INT UNSIGNED NOT NULL,
    CumulativeRecall DECIMAL(18, 12) NOT NULL,
    PRIMARY KEY (Scenario, Model, ScoreDecileHighToLow)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_calibration_bins (
    Scenario VARCHAR(40) NOT NULL,
    Model VARCHAR(40) NOT NULL,
    ScoreBinLowToHigh TINYINT UNSIGNED NOT NULL,
    Sessions INT UNSIGNED NOT NULL,
    Purchases INT UNSIGNED NOT NULL,
    MeanScore DECIMAL(18, 12) NOT NULL,
    ObservedConversionRate DECIMAL(18, 12) NOT NULL,
    CalibrationGapObservedMinusScore DECIMAL(18, 12) NOT NULL,
    PRIMARY KEY (Scenario, Model, ScoreBinLowToHigh)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_metric_ci (
    Scenario VARCHAR(40) NOT NULL,
    ScenarioName VARCHAR(80) NOT NULL,
    Metric VARCHAR(40) NOT NULL,
    Lower95 DECIMAL(18, 12) NOT NULL,
    Upper95 DECIMAL(18, 12) NOT NULL,
    BootstrapIterations INT UNSIGNED NOT NULL,
    PRIMARY KEY (Scenario, Metric)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- statement-break

CREATE TABLE IF NOT EXISTS model_feature_availability (
    FeatureGroup VARCHAR(100) NOT NULL,
    Fields VARCHAR(500) NOT NULL,
    Availability VARCHAR(300) NOT NULL,
    AllowedForRealTimeEarlyIntervention VARCHAR(20) NOT NULL,
    Decision VARCHAR(1000) NOT NULL,
    PRIMARY KEY (FeatureGroup)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
