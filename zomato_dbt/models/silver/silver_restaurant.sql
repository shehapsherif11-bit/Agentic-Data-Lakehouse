{{ config(
    materialized='table'
) }}

SELECT 
    -- تأمين الـ id وتحويله لرقم
    TRY_CAST(id AS INT) AS restaurant_id,
    
    TRIM(name) AS restaurant_name,
    TRIM(city) AS city,
    
    -- السحر هنا: أي '--' هتتحول لـ NULL بأمان تام
    TRY_CAST(rating AS DOUBLE) AS rating,
    TRY_CAST(rating_count AS INT) AS rating_count,
    TRY_CAST(cost AS DOUBLE) AS cost,
    
    TRIM(cuisine) AS cuisine,
    TRIM(lic_no) AS lic_no,
    TRIM(link) AS link,
    TRIM(address) AS address,
    TRIM(menu) AS menu

    -- لاحظ إننا مكتبناش _c0 خالص، فكده اتخلصنا منه نهائياً

FROM {{ source('zomato_bronze', 'bronze_restaurant') }}

-- تنظيف أي تكرار للمطعم الواحد
QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) = 1