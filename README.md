# Bot_Clicks_Detection

The Ongage Activity Anomaly Detection Interface reads click-activity records from SQL Server, normalizes and engineers time-based and user/IP features, applies an Isolation Forest to flag anomalous sessions, then writes both the raw and anomaly-flagged datasets back into Delta and into SQL Server for downstream analysis.

 [SQL Server: DataOps_Data]
           ↓ (JDBC read)
    [Spark DataFrame: ongage_activity]
           ↓ (filter & enrich)
    [df_clicks with features]
           ↓ (VectorAssembler)
    [features_vector_df]
           ↓ (toPandas → scikit-learn)
    [IsolationForest.predict]
           ↓ (join anomaly flags)
    [df_clicks_with_anomaly]
           ↓ (Delta write)
 [pocn_data.silver.ongage_activity_anomaly_latest]
           ↓ (JDBC write)
 [SQL Server: ongage_activity_anomaly]
