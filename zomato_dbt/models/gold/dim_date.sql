{{ config(
    materialized='table',
    schema='zomato_gold'
) }}

-- بنستخرج كل التواريخ الفريدة الموجودة في أوردرات السيلفر عشان نبني منها جدول التاريخ
WITH distinct_dates AS (
    SELECT DISTINCT order_date AS full_date
    FROM {{ ref('silver_orders') }}
    WHERE order_date IS NOT NULL
)

SELECT 
    -- مفتاح التاريخ بصيغة رقمية (مثال: 20251201) لربط سهل وسريع
    CAST(DATE_FORMAT(full_date, 'yyyyMMdd') AS INT) AS date_id,
    full_date,
    
    -- الأجزاء الزمنية الأساسية
    YEAR(full_date) AS year,
    QUARTER(full_date) AS quarter,
    CONCAT('Q', QUARTER(full_date)) AS quarter_name,
    MONTH(full_date) AS month_number,
    DATE_FORMAT(full_date, 'MMMM') AS month_name,
    DATE_FORMAT(full_date, 'MMM') AS month_short,
    DAY(full_date) AS day_number,
    
    -- أسماء الأيام للتحليلات الأسبوعية
    DATE_FORMAT(full_date, 'EEEE') AS day_name,
    
    -- معرفة هل اليوم ويك إند ولا يوم عمل (عشان تحليل سلوك الزبائن)
    CASE 
        WHEN DAYOFWEEK(full_date) IN (1, 7) THEN 'Weekend' -- (يختلف حسب أيام الإجازة، عادة السبت والأحد أو الجمعة والسبت)
        ELSE 'Weekday'
    END AS day_type

FROM distinct_dates
ORDER BY full_date ASC