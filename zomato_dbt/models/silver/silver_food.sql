{{ config(
    materialized='table'
) }}

SELECT 
    -- هنسيبه نص (String) زي ما هو عشان نحافظ على العلاقات مع الجداول التانية
    TRIM(f_id) AS food_id,
    
    -- تنظيف أسماء الأكلات من أي مسافات
    TRIM(item) AS item_name,
    
    -- تنظيف فئة الأكل (نباتي/غير نباتي)
    TRIM(veg_or_non_veg) AS veg_or_non_veg

FROM {{ source('zomato_bronze', 'bronze_food') }}
-- لو اسم الجدول في البرونز مختلف عندك تأكد إنك تعدله

-- تأمين الجدول ضد أي أكلات متكررة بنفس الـ ID
QUALIFY ROW_NUMBER() OVER (PARTITION BY f_id ORDER BY f_id) = 1