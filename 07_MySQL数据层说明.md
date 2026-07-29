# MySQL 数据层说明

## 1. 本阶段交付内容

本阶段把已经完成的数据清洗、字段工程和模型评估结果装入 MySQL，供 Navicat、SQL 查询和 Power BI 使用。

数据库默认名称为 `online_shoppers_analytics`，包含：

| 对象 | 用途 |
| --- | --- |
| `fact_sessions` | 12,330 条会话行为与购买结果 |
| `fact_model_scores` | 每条会话的样本外倾向分数及评分来源 |
| `dim_month` | 1-12 月排序维表 |
| `model_evaluation` | 两个优胜模型的测试集指标 |
| `model_operating_points` | Top 5%-30% 触达档位的 Precision、Recall 和 Lift |
| `model_decile_lift` | 测试集十分位 Lift |
| `model_calibration_bins` | 倾向分数校准诊断分箱 |
| `model_metric_ci` | Bootstrap 95% 置信区间 |
| `model_feature_availability` | 特征实际可用时点与部署边界 |

模型分数统一称为“倾向分数”，不称为经过校准的购买概率。`PageValues` 模型仍然只是回顾性基准。

## 2. 文件结构

| 文件 | 作用 |
| --- | --- |
| `07_MySQL数据层.py` | 校验 CSV、创建数据库、建表、装载数据、创建视图并核验结果 |
| `sql/07_mysql_schema.sql` | MySQL 8.0 表、约束和索引 |
| `sql/07_mysql_views.sql` | Power BI 与业务分析视图 |
| `sql/07_mysql_validation.sql` | 可在 Navicat 中手动执行的只读核验 SQL |
| `requirements-sql.txt` | Python 装载脚本所需驱动 |

## 3. 第一次运行

先在项目目录安装 Python 驱动：

```powershell
python -m pip install -r requirements-sql.txt
```

然后运行：

```powershell
python 07_MySQL数据层.py --user root
```

脚本会提示输入 MySQL 密码，密码不会显示，也不会写入项目文件。默认连接：

- 主机：`localhost`
- 端口：`3306`
- 数据库：`online_shoppers_analytics`
- 用户：`root`

如果 Navicat 使用的不是 `root`，把命令中的用户名换成 Navicat 连接所用用户。也可以显式指定连接参数：

```powershell
python 07_MySQL数据层.py --host localhost --port 3306 --user your_user --database online_shoppers_analytics
```

需要自动化运行时，可以临时通过环境变量传入密码：

```powershell
$env:MYSQL_PASSWORD = "你的密码"
python 07_MySQL数据层.py --user root
Remove-Item Env:MYSQL_PASSWORD
```

不要把真实密码写进 Python、SQL、PBIX 或版本控制文件。

## 4. 重复运行规则

脚本可以重复运行：

1. 源 CSV 会先进行字段、类型、`SessionID`、目标变量和模型评分覆盖检查。
2. 已有表结构和数据库会保留。
3. 八张数据表会在同一事务中删除旧数据并重新装载。
4. 任一装载步骤失败都会回滚数据刷新。
5. 视图会使用 `CREATE OR REPLACE VIEW` 更新。

只建表和视图、不刷新数据时使用：

```powershell
python 07_MySQL数据层.py --user root --schema-only
```

## 5. 主要分析视图

| 视图 | 用途 |
| --- | --- |
| `vw_session_analysis` | 保留 27 个分析字段并合并样本外模型分数，适合作为 Power BI 主表 |
| `vw_conversion_overview` | 网站整体转化 KPI |
| `vw_visitor_performance` | 新访客、回访访客和 Other 对比 |
| `vw_month_performance` | 含完整月份排序的月度表现 |
| `vw_weekend_performance` | 周末与工作日表现 |
| `vw_traffic_performance` | 渠道量级、转化率及高流量低转化标签 |
| `vw_device_performance` | 操作系统和浏览器表现 |
| `vw_behavior_performance` | 浏览深度、停留时间、页面访问、跳出和退出分组 |
| `vw_growth_opportunities` | 未购买会话明细及 P1-P5 机会优先级 |
| `vw_growth_opportunity_summary` | 未购买机会群体汇总 |

`vw_growth_opportunities` 的优先级是业务筛选规则，不是因果结论：

| 优先级 | 定义 |
| --- | --- |
| P1 | 样本外行为倾向分数位于未购买会话 Top 10% |
| P2 | 高浏览或高停留，并且是回访访客 |
| P3 | 高浏览或高停留，但不满足 P2 |
| P4 | 其余回访未购买会话 |
| P5 | 其余未购买会话 |

其中“高浏览”为 `ProductRelated >= 21`，“高停留”为 `ProductDurationGroup = 高`。

## 6. Power BI 连接建议

Python 的 `mysql-connector-python` 只服务于数据装载，不能替代 Power BI 所需的 Oracle MySQL Connector/NET。请先通过电脑上的 MySQL Installer 补装兼容的 Connector/NET，然后在 Power BI 中选择：

1. 获取数据。
2. MySQL 数据库。
3. 服务器填写 `localhost:3306`。
4. 数据库填写 `online_shoppers_analytics`。
5. 本项目使用 Import 模式。

推荐导入：

- 主分析表：`vw_session_analysis`
- 月份维表：`dim_month`
- 模型页：`model_evaluation`、`model_operating_points`、`model_decile_lift`、`model_metric_ci`

如果已经使用 CSV 建好报告，可以在 Power Query 中把主查询的数据源替换为 `vw_session_analysis`。原 27 个业务字段名称保持一致，新增的模型字段不会破坏原图表。

Power BI 中仍需设置排序：

- `Month` 按 `MonthOrder` 排序。
- `ProductDepthGroup` 按 `ProductDepthOrder` 排序。
- `ProductDurationGroup` 按 `ProductDurationOrder` 排序。

稳定的购买指标建议使用 `RevenueFlag`：

```DAX
会话数量 = COUNTROWS(vw_session_analysis)

购买会话数 = SUM(vw_session_analysis[RevenueFlag])

购买转化率 = DIVIDE([购买会话数], [会话数量])
```

## 7. 数据核验标准

成功装载后，脚本应输出：

- `fact_sessions`：12,330 行。
- `fact_model_scores`：12,330 行。
- 购买会话：1,908 行。
- 整体购买转化率：15.47%。
- `Revenue` 与 `RevenueFlag` 不一致：0 行。
- 缺少样本外模型评分：0 行。

还可以在 Navicat 中选择 `online_shoppers_analytics` 数据库后执行 `sql/07_mysql_validation.sql` 做人工复核。
