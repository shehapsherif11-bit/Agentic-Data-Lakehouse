{{ config(
    materialized='table'
) }}

SELECT 
    -- تأمين الـ ID
    TRY_CAST(user_id AS INT) AS user_id,
    
    -- تنظيف النصوص
    TRIM(name) AS name,
    lower(email) AS email,
    -- لاحظ إننا شيلنا الـ password خالص 
    
    -- تأمين العمر وحجم العائلة كأرقام
    TRY_CAST(Age AS INT) AS age,
    TRY_CAST(Family_size AS INT) AS family_size,
    
    -- تنظيف باقي الأعمدة النصية
    TRIM(Gender) AS gender,
    TRIM(Marital_Status) AS marital_status,
    TRIM(Occupation) AS occupation,
    TRIM(Monthly_Income) AS monthly_income,
    TRIM(Educational_Qualifications) AS educational_qualifications

FROM {{ source('zomato_bronze', 'bronze_users') }}
-- تأكد إن اسم الجدول في الـ source صح (لو اسمه users بس، عدلها)

-- إزالة أي يوزر متكرر بناءً على الـ user_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY user_id) = 1