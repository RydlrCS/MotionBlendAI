
  
    

    create or replace table `motionblend-ai`.`RAW_DEV_marts`.`mart_blend_snn_metrics`
      
    
    

    
    OPTIONS()
    as (
      -- models/marts/mart_blend_snn_metrics.sql
-- Final metrics for blend_snn quality analysis

WITH blend_base AS (
    SELECT
        blend_id,
        left_motion_id,
        right_motion_id,
        blend_ratio,
        transition_start_frame,
        transition_end_frame
    FROM `motionblend-ai`.`RAW_DEV_staging`.`stg_blend_snn`
),

-- Placeholder for computed metrics
-- In production, these would come from UDFs or pre-computed feature tables
metrics_placeholder AS (
    SELECT
        blend_id,
        -- Quality metrics (0-1 scale)
        0.19 AS fid,
        0.97 AS coverage,
        1.56 AS gdiv,
        1.44 AS ldiv,
        0.60 AS inter_div,
        0.38 AS intra_div,
        
        -- Velocity/acceleration stats
        0.3245 AS l2_velocity_mean,
        0.0892 AS l2_acceleration_mean,
        0.1234 AS transition_smoothness,
        
        -- Quality score (derived)
        0.85 AS quality_score,
        'good' AS quality_category
    FROM blend_base
)

SELECT
    b.*,
    m.fid,
    m.coverage,
    m.gdiv,
    m.ldiv,
    m.inter_div,
    m.intra_div,
    m.l2_velocity_mean,
    m.l2_acceleration_mean,
    m.transition_smoothness,
    m.quality_score,
    m.quality_category,
    CURRENT_TIMESTAMP() AS computed_at
FROM blend_base b
JOIN metrics_placeholder m USING (blend_id)
    );
  