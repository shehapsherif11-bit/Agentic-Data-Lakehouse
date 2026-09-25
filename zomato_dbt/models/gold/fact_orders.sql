{{ config(
    materialized='table',
    schema='zomato_gold'
) }}

WITH orders AS (
    SELECT * FROM {{ ref('silver_orders') }}
)

SELECT 
    -- 1. Keys for relationships
    order_id,
    user_id,
    restaurant_id,
    
    -- 2. Date Key for dim_date linkage
    CAST(DATE_FORMAT(order_date, 'yyyyMMdd') AS INT) AS date_id,
    order_timestamp,
    
    -- 3. Order details
    restaurant_city,
    order_status,
    
    -- 4. Measures (The Numbers)
    items_count,
    sales_qty,
    subtotal,
    discount,
    delivery_fee,
    gst,
    sales_amount,
    customer_rating,
    delivery_time_min
    
FROM orders