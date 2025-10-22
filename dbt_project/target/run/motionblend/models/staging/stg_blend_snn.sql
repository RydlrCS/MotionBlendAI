

  create or replace view `motionblend-ai`.`RAW_DEV_staging`.`stg_blend_snn`
  OPTIONS()
  as -- models/staging/stg_blend_snn.sql
-- Staging layer for blend_snn (Smooth Neural Network blends)

WITH source AS (
    SELECT * FROM `motionblend-ai`.`RAW_DEV`.`blend_snn`
),

renamed AS (
    SELECT
        id AS blend_id,
        left_motion_id,
        right_motion_id,
        blend_ratio,
        transition_start_frame,
        transition_end_frame,
        created_at,
        updated_at
    FROM source
)

SELECT * FROM renamed;

