-- Run these read-only checks after loading data.

SELECT 'fact_sessions' AS ObjectName, COUNT(*) AS RowCount FROM fact_sessions
UNION ALL
SELECT 'fact_model_scores', COUNT(*) FROM fact_model_scores
UNION ALL
SELECT 'model_evaluation', COUNT(*) FROM model_evaluation
UNION ALL
SELECT 'model_operating_points', COUNT(*) FROM model_operating_points
UNION ALL
SELECT 'model_decile_lift', COUNT(*) FROM model_decile_lift
UNION ALL
SELECT 'model_calibration_bins', COUNT(*) FROM model_calibration_bins
UNION ALL
SELECT 'model_metric_ci', COUNT(*) FROM model_metric_ci
UNION ALL
SELECT 'model_feature_availability', COUNT(*) FROM model_feature_availability;

SELECT * FROM vw_conversion_overview;

SELECT
    SUM(Revenue <> RevenueFlag) AS RevenueFlagMismatchCount,
    SUM(MonthOrder <> (SELECT MonthOrder FROM dim_month WHERE MonthCode = fact_sessions.Month))
        AS MonthOrderMismatchCount
FROM fact_sessions;

SELECT COUNT(*) AS MissingModelScoreCount
FROM fact_sessions AS s
LEFT JOIN fact_model_scores AS ms ON s.SessionID = ms.SessionID
WHERE ms.SessionID IS NULL;

SELECT COUNT(*) AS ModelRevenueMismatchCount
FROM fact_sessions AS s
JOIN fact_model_scores AS ms ON s.SessionID = ms.SessionID
WHERE s.RevenueFlag <> ms.RevenueFlag;

SELECT * FROM vw_growth_opportunity_summary ORDER BY OpportunityPriority;
