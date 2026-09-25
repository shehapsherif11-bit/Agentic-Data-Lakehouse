{{ config(
    materialized='table',
    schema='zomato_gold'  
) }}

-- 1. سحب الداتا من طبقة السيلفر باستخدام دالة ref()
WITH menu AS (
    SELECT * FROM {{ ref('silver_menu') }}
),

food AS (
    SELECT * FROM {{ ref('silver_food') }}
)

-- 2. دمج الجدولين عشان نطلع بجدول Dimension نهائي ونظيف
SELECT 
    m.menu_id,
    m.restaurant_id,
    m.food_id,
    f.item_name,
    f.veg_or_non_veg,
    m.cuisine,
    m.price
FROM menu m
LEFT JOIN food f 
    ON m.food_id = f.food_id