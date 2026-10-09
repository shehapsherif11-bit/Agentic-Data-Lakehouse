{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_item_id',
    file_format='delta',
    on_schema_change='sync_all_columns'
) }}

WITH src AS (
    SELECT
        -- تأمين الـ IDs الأساسية
        TRY_CAST(order_item_id AS INT) AS order_item_id,
        TRY_CAST(order_id AS INT) AS order_id,

        -- توحيد أسماء الـ IDs اللي هتربطنا بالجداول التانية
        TRY_CAST(r_id AS INT) AS restaurant_id,
        TRIM(f_id) AS food_id,

        -- تظبيط الأرقام والحسابات
        TRY_CAST(price AS DOUBLE) AS price,
        TRY_CAST(quantity AS INT) AS quantity,
        TRY_CAST(line_amount AS DOUBLE) AS line_amount

    FROM {{ source('zomato_bronze', 'bronze_order_items') }}
)

SELECT s.*
FROM src s
{% if is_incremental() %}
-- Incremental: order items are immutable and carry no timestamp, so load only keys not yet in the target.
LEFT ANTI JOIN {{ this }} t ON s.order_item_id = t.order_item_id
{% endif %}

-- منع التكرار بناءً على الـ order_item_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY s.order_item_id ORDER BY s.order_item_id) = 1
