{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_id',
    file_format='delta',
    on_schema_change='sync_all_columns'
) }}

SELECT 
    -- 1. تأمين الـ IDs وتوحيد الأسماء
    TRY_CAST(order_id AS INT) AS order_id,
    TRY_CAST(user_id AS INT) AS user_id,
    TRY_CAST(r_id AS INT) AS restaurant_id, -- غيرنا اسمه عشان الـ Joins
    
    -- 2. تأمين التواريخ والوقت
    TRY_CAST(order_timestamp AS TIMESTAMP) AS order_timestamp,
    TRY_CAST(order_date AS DATE) AS order_date,
    
    -- 3. تنظيف النصوص
    TRIM(restaurant_city) AS restaurant_city,
    TRIM(cuisine) AS cuisine,
    TRIM(currency) AS currency,
    TRIM(payment_method) AS payment_method,
    TRIM(order_status) AS order_status,
    
    -- 4. تأمين الكميات
    TRY_CAST(items_count AS INT) AS items_count,
    TRY_CAST(sales_qty AS INT) AS sales_qty,
    
    -- 5. تأمين الحسابات المالية (الفلوس)
    TRY_CAST(subtotal AS DOUBLE) AS subtotal,
    TRY_CAST(discount AS DOUBLE) AS discount,
    TRY_CAST(delivery_fee AS DOUBLE) AS delivery_fee,
    TRY_CAST(gst AS DOUBLE) AS gst,
    TRY_CAST(sales_amount AS DOUBLE) AS sales_amount,
    
    -- 6. تأمين التقييم ووقت التوصيل
    TRY_CAST(customer_rating AS DOUBLE) AS customer_rating,
    TRY_CAST(delivery_time_min AS INT) AS delivery_time_min

FROM {{ source('zomato_bronze', 'bronze_orders') }}

{% if is_incremental() %}
-- Incremental: only look at rows newer than what is already loaded. The 3-day lookback re-reads a small
-- window so late-arriving rows are picked up; MERGE on order_id makes the overlap idempotent.
WHERE TRY_CAST(order_timestamp AS TIMESTAMP) >= (
    SELECT COALESCE(MAX(order_timestamp), TIMESTAMP '1900-01-01') - INTERVAL 3 DAYS FROM {{ this }}
)
{% endif %}

-- منع تكرار الطلبات (لو الطلب متكرر، هناخد أحدث نسخة منه بناءً على وقت الطلب)
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_timestamp DESC) = 1
