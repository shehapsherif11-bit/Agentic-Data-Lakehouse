{{ config(
    materialized='table'
) }}

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
-- لو اسم الجدول في البرونز مختلف عندك تأكد إنك تعدله

-- منع التكرار بناءً على الـ order_item_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_item_id ORDER BY order_item_id) = 1