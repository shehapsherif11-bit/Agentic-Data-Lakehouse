{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_item_id',
    file_format='delta',
    on_schema_change='sync_all_columns',
    schema='zomato_gold'
) }}

WITH items AS (
    SELECT i.* FROM {{ ref('silver_order_items') }} i
    {% if is_incremental() %}
    -- Only items not yet in the fact table (order items are immutable); anti-join scales better than NOT IN.
    LEFT ANTI JOIN {{ this }} t ON i.order_item_id = t.order_item_id
    {% endif %}
)

SELECT
    -- 1. Keys for relationships
    order_item_id,  -- merge key for incremental loads
    order_id,
    food_id,

    -- 2. Measures (The Numbers)
    quantity,
    price,
    line_amount

FROM items
