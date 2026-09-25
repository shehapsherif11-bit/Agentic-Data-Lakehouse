{{ config(
    materialized='table',
    schema='zomato_gold'  
) }}

WITH users AS (
    SELECT * FROM {{ ref('silver_users') }}
)

SELECT 
    user_id,
    name,
    email,
    age,
    
    -- 1. استخراج فئات عمرية (Age Grouping)
    CASE 
        WHEN age BETWEEN 18 AND 24 THEN 'Gen Z'
        WHEN age BETWEEN 25 AND 34 THEN 'Millennials'
        WHEN age BETWEEN 35 AND 44 THEN 'Adults'
        WHEN age >= 45 THEN 'Seniors'
        ELSE 'Unknown'
    END AS age_group,
    
    gender,
    marital_status,
    occupation,
    monthly_income,
    educational_qualifications,
    family_size,
    
    -- 2. استخراج شريحة العائلة (Family Segmentation)
    CASE 
        WHEN family_size = 1 THEN 'Single'
        WHEN family_size = 2 THEN 'Couple'
        WHEN family_size BETWEEN 3 AND 4 THEN 'Small Family'
        WHEN family_size >= 5 THEN 'Large Family'
        ELSE 'Unknown'
    END AS family_segment

FROM users