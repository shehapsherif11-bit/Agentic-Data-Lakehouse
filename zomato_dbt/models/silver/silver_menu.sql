{{ config(
    materialized='table'
) }}

SELECT 
    -- الحفاظ على الـ IDs اللي فيها حروف كنصوص (String)
    TRIM(menu_id) AS menu_id,
    TRIM(f_id) AS food_id,
    
    -- تحويل رقم المطعم لـ INT وتوحيد اسمه عشان يربط صح مع جدول المطاعم
    TRY_CAST(r_id AS INT) AS restaurant_id,
    
    -- تنظيف أسماء المطابخ/الأكلات
    TRIM(cuisine) AS cuisine,
    
    -- زي ما إنت قولت بالظبط: تحويل السعر لـ DOUBLE بأمان
    TRY_CAST(price AS DOUBLE) AS price

FROM {{ source('zomato_bronze', 'bronze_menu') }}
-- لو اسم الجدول في البرونز مختلف عندك تأكد إنك تعدله

-- هنا هنمنع التكرار بناءً على الوجبة جوه المنيو
QUALIFY ROW_NUMBER() OVER (PARTITION BY menu_id, f_id ORDER BY menu_id) = 1