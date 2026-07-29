-- Power BI and validation views.
-- Conversion rates are decimals in the range 0-1, not percentage points.

CREATE OR REPLACE VIEW vw_session_analysis AS
SELECT
    s.SessionID,
    s.Administrative,
    s.Administrative_Duration,
    s.Informational,
    s.Informational_Duration,
    s.ProductRelated,
    s.ProductRelated_Duration,
    s.BounceRates,
    s.ExitRates,
    s.PageValues,
    s.SpecialDay,
    s.Month,
    s.OperatingSystems,
    s.Browser,
    s.Region,
    s.TrafficType,
    s.VisitorType,
    s.Weekend,
    s.Revenue,
    s.RevenueFlag,
    s.MonthOrder,
    s.ProductDepthGroup,
    s.ProductDepthOrder,
    s.ProductDurationGroup,
    s.ProductDurationOrder,
    s.HighBounceSession,
    s.HighExitSession,
    ms.BehaviorExclPageValuesOutOfSampleScore AS BehaviorPropensityScore,
    ms.BehaviorExclPageValuesScoreSource AS BehaviorPropensityScoreSource,
    ms.PageValuesBenchmarkOutOfSampleScore AS PageValuesRetrospectiveBenchmarkScore,
    ms.PageValuesBenchmarkScoreSource AS PageValuesBenchmarkScoreSource,
    ms.BehaviorExclPageValuesPercentileAmongUnpurchased AS PropensityPercentileAmongUnpurchased,
    ms.HistoricalHighPotentialTop10Percent
FROM fact_sessions AS s
LEFT JOIN fact_model_scores AS ms
    ON s.SessionID = ms.SessionID;

-- statement-break

CREATE OR REPLACE VIEW vw_conversion_overview AS
SELECT
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate,
    ROUND(AVG(VisitorType = 'New_Visitor'), 6) AS NewVisitorShare,
    ROUND(
        SUM(CASE WHEN VisitorType = 'Returning_Visitor' THEN RevenueFlag ELSE 0 END)
        / NULLIF(SUM(VisitorType = 'Returning_Visitor'), 0),
        6
    ) AS ReturningVisitorConversionRate,
    ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
    ROUND(AVG(ProductRelated_Duration), 4) AS AverageProductPageDuration,
    ROUND(AVG(BounceRates), 6) AS AverageBounceRate,
    ROUND(AVG(ExitRates), 6) AS AverageExitRate
FROM fact_sessions;

-- statement-break

CREATE OR REPLACE VIEW vw_visitor_performance AS
SELECT
    VisitorType,
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate,
    ROUND(COUNT(*) / (SELECT COUNT(*) FROM fact_sessions), 6) AS SessionShare,
    ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
    ROUND(AVG(ProductRelated_Duration), 4) AS AverageProductPageDuration,
    ROUND(AVG(BounceRates), 6) AS AverageBounceRate,
    ROUND(AVG(ExitRates), 6) AS AverageExitRate
FROM fact_sessions
GROUP BY VisitorType;

-- statement-break

CREATE OR REPLACE VIEW vw_month_performance AS
SELECT
    m.MonthOrder,
    m.MonthCode AS Month,
    m.MonthNameChinese,
    COUNT(s.SessionID) AS SessionCount,
    COALESCE(SUM(s.RevenueFlag), 0) AS PurchaseSessionCount,
    CASE
        WHEN COUNT(s.SessionID) = 0 THEN NULL
        ELSE ROUND(AVG(s.RevenueFlag), 6)
    END AS PurchaseConversionRate,
    ROUND(AVG(s.ProductRelated), 4) AS AverageProductPageViews
FROM dim_month AS m
LEFT JOIN fact_sessions AS s
    ON m.MonthCode = s.Month
GROUP BY m.MonthOrder, m.MonthCode, m.MonthNameChinese;

-- statement-break

CREATE OR REPLACE VIEW vw_weekend_performance AS
SELECT
    Weekend,
    CASE WHEN Weekend = 1 THEN '周末' ELSE '工作日' END AS DayType,
    Weekend + 1 AS DayTypeOrder,
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate,
    ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
    ROUND(AVG(ProductRelated_Duration), 4) AS AverageProductPageDuration,
    ROUND(AVG(BounceRates), 6) AS AverageBounceRate,
    ROUND(AVG(ExitRates), 6) AS AverageExitRate
FROM fact_sessions
GROUP BY Weekend;

-- statement-break

CREATE OR REPLACE VIEW vw_traffic_performance AS
SELECT
    channel_stats.TrafficType,
    channel_stats.SessionCount,
    channel_stats.PurchaseSessionCount,
    channel_stats.PurchaseConversionRate,
    ROUND(channel_stats.SessionCount / benchmarks.TotalSessions, 6) AS SessionShare,
    channel_stats.AverageProductPageViews,
    channel_stats.AverageBounceRate,
    channel_stats.AverageExitRate,
    ROUND(benchmarks.OverallConversionRate, 6) AS OverallConversionRate,
    CASE
        WHEN channel_stats.SessionCount >= benchmarks.AverageChannelSessions
             AND channel_stats.PurchaseConversionRate < benchmarks.OverallConversionRate
            THEN '高流量低转化'
        WHEN channel_stats.SessionCount >= benchmarks.AverageChannelSessions
             AND channel_stats.PurchaseConversionRate >= benchmarks.OverallConversionRate
            THEN '高流量高转化'
        WHEN channel_stats.PurchaseConversionRate >= benchmarks.OverallConversionRate
            THEN '低流量高转化'
        ELSE '低流量低转化'
    END AS ChannelOpportunityType
FROM (
    SELECT
        TrafficType,
        COUNT(*) AS SessionCount,
        SUM(RevenueFlag) AS PurchaseSessionCount,
        AVG(RevenueFlag) AS PurchaseConversionRate,
        ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
        ROUND(AVG(BounceRates), 6) AS AverageBounceRate,
        ROUND(AVG(ExitRates), 6) AS AverageExitRate
    FROM fact_sessions
    GROUP BY TrafficType
) AS channel_stats
CROSS JOIN (
    SELECT
        COUNT(*) AS TotalSessions,
        COUNT(*) / COUNT(DISTINCT TrafficType) AS AverageChannelSessions,
        AVG(RevenueFlag) AS OverallConversionRate
    FROM fact_sessions
) AS benchmarks;

-- statement-break

CREATE OR REPLACE VIEW vw_device_performance AS
SELECT
    'OperatingSystem' AS DeviceDimension,
    CAST(OperatingSystems AS CHAR) AS DeviceValue,
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate
FROM fact_sessions
GROUP BY OperatingSystems
UNION ALL
SELECT
    'Browser' AS DeviceDimension,
    CAST(Browser AS CHAR) AS DeviceValue,
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate
FROM fact_sessions
GROUP BY Browser;

-- statement-break

CREATE OR REPLACE VIEW vw_behavior_performance AS
SELECT
    'ProductDepth' AS BehaviorDimension,
    ProductDepthGroup AS Segment,
    ProductDepthOrder AS SegmentOrder,
    COUNT(*) AS SessionCount,
    SUM(RevenueFlag) AS PurchaseSessionCount,
    ROUND(AVG(RevenueFlag), 6) AS PurchaseConversionRate,
    ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
    ROUND(AVG(ProductRelated_Duration), 4) AS AverageProductPageDuration,
    ROUND(AVG(BounceRates), 6) AS AverageBounceRate,
    ROUND(AVG(ExitRates), 6) AS AverageExitRate
FROM fact_sessions
GROUP BY ProductDepthGroup, ProductDepthOrder
UNION ALL
SELECT
    'ProductDuration',
    ProductDurationGroup,
    ProductDurationOrder,
    COUNT(*),
    SUM(RevenueFlag),
    ROUND(AVG(RevenueFlag), 6),
    ROUND(AVG(ProductRelated), 4),
    ROUND(AVG(ProductRelated_Duration), 4),
    ROUND(AVG(BounceRates), 6),
    ROUND(AVG(ExitRates), 6)
FROM fact_sessions
GROUP BY ProductDurationGroup, ProductDurationOrder
UNION ALL
SELECT
    'HighBounce',
    CASE WHEN HighBounceSession = 1 THEN '高跳出' ELSE '非高跳出' END,
    HighBounceSession + 1,
    COUNT(*),
    SUM(RevenueFlag),
    ROUND(AVG(RevenueFlag), 6),
    ROUND(AVG(ProductRelated), 4),
    ROUND(AVG(ProductRelated_Duration), 4),
    ROUND(AVG(BounceRates), 6),
    ROUND(AVG(ExitRates), 6)
FROM fact_sessions
GROUP BY HighBounceSession
UNION ALL
SELECT
    'HighExit',
    CASE WHEN HighExitSession = 1 THEN '高退出' ELSE '非高退出' END,
    HighExitSession + 1,
    COUNT(*),
    SUM(RevenueFlag),
    ROUND(AVG(RevenueFlag), 6),
    ROUND(AVG(ProductRelated), 4),
    ROUND(AVG(ProductRelated_Duration), 4),
    ROUND(AVG(BounceRates), 6),
    ROUND(AVG(ExitRates), 6)
FROM fact_sessions
GROUP BY HighExitSession
UNION ALL
SELECT
    'InformationalPage',
    CASE WHEN Informational > 0 THEN '访问过信息页' ELSE '未访问信息页' END,
    (Informational > 0) + 1,
    COUNT(*),
    SUM(RevenueFlag),
    ROUND(AVG(RevenueFlag), 6),
    ROUND(AVG(ProductRelated), 4),
    ROUND(AVG(ProductRelated_Duration), 4),
    ROUND(AVG(BounceRates), 6),
    ROUND(AVG(ExitRates), 6)
FROM fact_sessions
GROUP BY (Informational > 0)
UNION ALL
SELECT
    'AdministrativePage',
    CASE WHEN Administrative > 0 THEN '访问过管理页' ELSE '未访问管理页' END,
    (Administrative > 0) + 1,
    COUNT(*),
    SUM(RevenueFlag),
    ROUND(AVG(RevenueFlag), 6),
    ROUND(AVG(ProductRelated), 4),
    ROUND(AVG(ProductRelated_Duration), 4),
    ROUND(AVG(BounceRates), 6),
    ROUND(AVG(ExitRates), 6)
FROM fact_sessions
GROUP BY (Administrative > 0);

-- statement-break

CREATE OR REPLACE VIEW vw_growth_opportunities AS
SELECT
    s.SessionID,
    s.Month,
    s.MonthOrder,
    s.VisitorType,
    s.TrafficType,
    s.Weekend,
    s.ProductRelated,
    s.ProductRelated_Duration,
    s.ProductDepthGroup,
    s.ProductDepthOrder,
    s.ProductDurationGroup,
    s.ProductDurationOrder,
    s.BounceRates,
    s.ExitRates,
    s.HighBounceSession,
    s.HighExitSession,
    ms.BehaviorExclPageValuesOutOfSampleScore AS BehaviorPropensityScore,
    ms.BehaviorExclPageValuesScoreSource AS BehaviorPropensityScoreSource,
    ms.BehaviorExclPageValuesPercentileAmongUnpurchased AS PropensityPercentileAmongUnpurchased,
    COALESCE(ms.HistoricalHighPotentialTop10Percent, 0) AS HistoricalHighPotentialTop10Percent,
    (s.ProductRelated >= 21) AS IsHighProductBrowsing,
    (s.ProductDurationGroup = '高') AS IsHighProductDuration,
    (s.VisitorType = 'Returning_Visitor') AS IsReturningVisitor,
    CASE
        WHEN COALESCE(ms.HistoricalHighPotentialTop10Percent, 0) = 1
            THEN 'P1 历史高潜未购买'
        WHEN s.VisitorType = 'Returning_Visitor'
             AND (s.ProductRelated >= 21 OR s.ProductDurationGroup = '高')
            THEN 'P2 高参与回访未购买'
        WHEN s.ProductRelated >= 21 OR s.ProductDurationGroup = '高'
            THEN 'P3 高参与未购买'
        WHEN s.VisitorType = 'Returning_Visitor'
            THEN 'P4 回访未购买'
        ELSE 'P5 其他未购买'
    END AS OpportunityPriority
FROM fact_sessions AS s
LEFT JOIN fact_model_scores AS ms
    ON s.SessionID = ms.SessionID
WHERE s.RevenueFlag = 0;

-- statement-break

CREATE OR REPLACE VIEW vw_growth_opportunity_summary AS
SELECT
    OpportunityPriority,
    COUNT(*) AS UnpurchasedSessionCount,
    ROUND(COUNT(*) / (SELECT COUNT(*) FROM vw_growth_opportunities), 6) AS ShareOfUnpurchasedSessions,
    ROUND(AVG(BehaviorPropensityScore), 6) AS AverageBehaviorPropensityScore,
    ROUND(AVG(ProductRelated), 4) AS AverageProductPageViews,
    ROUND(AVG(ProductRelated_Duration), 4) AS AverageProductPageDuration,
    ROUND(AVG(IsReturningVisitor), 6) AS ReturningVisitorShare,
    ROUND(AVG(HighBounceSession), 6) AS HighBounceShare,
    ROUND(AVG(HighExitSession), 6) AS HighExitShare
FROM vw_growth_opportunities
GROUP BY OpportunityPriority;
