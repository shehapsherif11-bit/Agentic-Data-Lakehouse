{{ config(
    materialized='table'
) }}

SELECT 
    -- تأمين الـ IDs وتحويلها لأرقام صحيحة
    TRY_CAST(review_id AS INT) AS review_id,
    TRY_CAST(order_id AS INT) AS order_id,
    TRY_CAST(user_id AS INT) AS user_id,
    TRY_CAST(restaurant_id AS INT) AS restaurant_id,
    
    -- تأمين التقييم كرقم
    TRY_CAST(rating AS INT) AS rating,
    
    -- تنظيف التعليق من المسافات الزايدة
    TRIM(comment) AS comment,
    
    -- تحويل التاريخ للنوع الصحيح (DATE)
    TRY_CAST(review_date AS DATE) AS review_date

FROM {{ source('zomato_bronze', 'bronze_reviews') }}

-- تنظيف أي تقييم متكرر بنفس الـ review_id (هناخد أحدث واحد لو فيه تكرار)
QUALIFY ROW_NUMBER() OVER (PARTITION BY review_id ORDER BY review_date DESC) = 1