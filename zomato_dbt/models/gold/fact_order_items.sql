{{ config(
    materialized='table',
    schema='zomato_gold'
) }}

WITH items AS (
    SELECT * FROM {{ ref('silver_order_items') }}
)

SELECT 
    -- 1. Keys for relationships
    order_id,
    food_id, 
    
    -- 2. Measures (The Numbers)
    quantity,
    price,
    line_amount
    
FROM items