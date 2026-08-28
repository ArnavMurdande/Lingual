/*=========================================================
  LOGISTICS LAB - COMPLETE SQL SCRIPT
=========================================================*/

-- Create Database and Schema
CREATE OR REPLACE DATABASE LOGISTICS_LAB;

USE DATABASE LOGISTICS_LAB;

CREATE OR REPLACE SCHEMA DATA;

USE SCHEMA DATA;

-- Create Deliveries Table
CREATE OR REPLACE TABLE DELIVERIES (
    ORDER_ID STRING,
    ORDER_DATE DATE,
    CITY STRING,
    VEHICLE_TYPE STRING,
    DISTANCE_KM NUMBER(8,2),
    DELIVERY_TIME_MIN INT,
    IS_ONTIME STRING,
    DRIVER_RATING NUMBER(3,1)
);

----------------------------------------------------------
-- Verification Queries (Run After Loading CSV)
----------------------------------------------------------

-- Check records loaded
SELECT COUNT(*) AS TOTAL_RECORDS
FROM DELIVERIES;

-- Preview data
SELECT *
FROM DELIVERIES
LIMIT 10;

----------------------------------------------------------
-- Query 1: Total Orders & Overall On-Time Rate
----------------------------------------------------------

SELECT
    COUNT(*) AS TOTAL_ORDERS,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN IS_ONTIME = 'Yes' THEN 1.0
                ELSE 0.0
            END
        ),
        1
    ) || '%' AS ONTIME_PERCENTAGE
FROM DELIVERIES;

----------------------------------------------------------
-- Query 2: Performance by City
----------------------------------------------------------

SELECT
    CITY,
    COUNT(*) AS TOTAL_DELIVERIES,
    ROUND(AVG(DELIVERY_TIME_MIN), 1) AS AVG_MINUTES,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN IS_ONTIME = 'Yes' THEN 1.0
                ELSE 0.0
            END
        ),
        1
    ) || '%' AS ONTIME_PCT
FROM DELIVERIES
GROUP BY CITY
ORDER BY ONTIME_PCT DESC;

----------------------------------------------------------
-- Query 3: Vehicle Type Performance
----------------------------------------------------------

SELECT
    VEHICLE_TYPE,
    ROUND(AVG(DELIVERY_TIME_MIN), 1) AS AVG_MINUTES,
    ROUND(AVG(DISTANCE_KM), 1) AS AVG_DISTANCE_KM,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN IS_ONTIME = 'Yes' THEN 1.0
                ELSE 0.0
            END
        ),
        1
    ) || '%' AS ONTIME_PCT
FROM DELIVERIES
GROUP BY VEHICLE_TYPE
ORDER BY AVG_MINUTES;

----------------------------------------------------------
-- Query 4: Top 10 Best-Rated Deliveries
----------------------------------------------------------

SELECT
    ORDER_ID,
    DRIVER_RATING,
    CITY,
    VEHICLE_TYPE,
    IS_ONTIME
FROM DELIVERIES
ORDER BY DRIVER_RATING DESC
LIMIT 10;

----------------------------------------------------------
-- Query 5: Daily Delivery Volume
----------------------------------------------------------

SELECT
    ORDER_DATE,
    COUNT(*) AS DELIVERIES_PER_DAY
FROM DELIVERIES
GROUP BY ORDER_DATE
ORDER BY ORDER_DATE;

