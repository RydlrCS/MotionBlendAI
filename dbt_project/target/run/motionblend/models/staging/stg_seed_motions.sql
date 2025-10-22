

  create or replace view `motionblend-ai`.`RAW_DEV_staging`.`stg_seed_motions`
  OPTIONS()
  as -- models/staging/stg_seed_motions.sql
-- Staging layer for raw seed motion files

WITH source AS (
    SELECT * FROM `motionblend-ai`.`RAW_DEV`.`seed_motions`
),

renamed AS (
    SELECT
        id,
        file_uri,
        skeleton_id,
        frames,
        fps,
        joints_count,
        created_at,
        updated_at
    FROM source
)

SELECT * FROM renamed;

