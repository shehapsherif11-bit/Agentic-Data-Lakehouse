{{ config(
    materialized='table',
    schema='zomato_gold'
) }}

WITH restaurants AS (
    SELECT * 
    FROM {{ ref('silver_restaurant') }}
    WHERE restaurant_id IS NOT NULL  -- السطر ده اللي هيشيل الـ 3 مطاعم المضروبين
)
SELECT 
    restaurant_id,
    restaurant_name,
    city,
    
    -- 1. Rating Category
    rating,
    CASE 
        WHEN rating >= 4.5 THEN 'Excellent'
        WHEN rating >= 4.0 AND rating < 4.5 THEN 'Very Good'
        WHEN rating >= 3.0 AND rating < 4.0 THEN 'Average'
        WHEN rating > 0.0 AND rating < 3.0 THEN 'Poor'
        ELSE 'Unrated'
    END AS rating_category,
    
    -- 2. Popularity Tier
    rating_count,
    CASE
        WHEN rating_count >= 1000 THEN 'High Traction'
        WHEN rating_count >= 500 THEN 'Medium Traction'
        WHEN rating_count > 0 THEN 'Low Traction'
        ELSE 'No Reviews'
    END AS popularity_tier,

    -- 3. Affordability Tier
    cost,
    CASE 
        WHEN cost >= 1000 THEN 'Premium'
        WHEN cost >= 500 THEN 'Mid-Range'
        WHEN cost > 0 THEN 'Budget Friendly'
        ELSE 'Unknown'
    END AS affordability_tier,
    
    -- 4. Cuisine Focus
    cuisine,
    CASE 
        WHEN cuisine LIKE '%,%' THEN 'Multi-Cuisine'
        ELSE 'Specialty Cuisine'
    END AS cuisine_focus

FROM restaurants