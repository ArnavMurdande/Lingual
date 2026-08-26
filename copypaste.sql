--Created Database
CREATE DATABASE IF NOT EXISTS ShopEasy_Profile;
USE ShopEasy_Profile;


--Created Table
CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    FullName VARCHAR(100),
    Email VARCHAR(120),
    Phone VARCHAR(20),
    City VARCHAR(50),
    Country VARCHAR(50)
);


--Inserted Data
INSERT INTO Customers VALUES
(1,'Amit  Sharma','amit@shop.com','9876543210','Delhi','India'),
(2,'Neha  Verma','neha@shop.com',' ','mumbai','india'),
(3,'Ravi Kumar',NULL,'9567843210','Bangalore','India'),
(4,'Amit  Sharma','amit@shop.com','9876543210','Delhi','India'),
(5,NULL,'john@shop.com','9999999999','Chennai','INDIA');

--Data Profiling
--Total Records
SELECT COUNT(*) AS TotalRecords
FROM Customers;


--Checked NULL and Blank Values
SELECT
    SUM(CASE WHEN FullName IS NULL OR TRIM(FullName) = '' THEN 1 ELSE 0 END) AS NullNames,
    SUM(CASE WHEN Email IS NULL OR TRIM(Email) = '' THEN 1 ELSE 0 END) AS NullEmails,
    SUM(CASE WHEN Phone IS NULL OR TRIM(Phone) = '' THEN 1 ELSE 0 END) AS NullPhones
FROM Customers;


--Checked Duplicate Customers
SELECT
    FullName,
    Email,
    COUNT(*) AS DupCount
FROM Customers
GROUP BY FullName, Email
HAVING COUNT(*) > 1;


--Checked Inconsistent Country Values
SELECT DISTINCT Country
FROM Customers;


--Data Cleaning
--Removed Extra Spaces and Standardize Text
UPDATE Customers
SET
    FullName = TRIM(FullName),
    City = CONCAT(
        UPPER(LEFT(TRIM(City), 1)),
        LOWER(SUBSTRING(TRIM(City), 2))
    ),
    Country = UPPER(TRIM(Country));


--Converted Blank Phone Numbers to NULL
UPDATE Customers
SET Phone = NULL
WHERE TRIM(Phone) = '';


--Replaced Missing Country Values
UPDATE Customers
SET Country = 'INDIA'
WHERE Country IS NULL;


--Replaced Missing Names
UPDATE Customers
SET FullName = 'Unknown Customer'
WHERE FullName IS NULL;


--Removed Duplicates (Keep Lowest CustomerID)
DELETE c1
FROM Customers c1
JOIN Customers c2
    ON c1.FullName = c2.FullName
   AND c1.Email = c2.Email
   AND c1.CustomerID > c2.CustomerID;



--Validation
--Viewed Cleaned Data
SELECT *
FROM Customers;


--Checked Phone Length Issues
SELECT *
FROM Customers
WHERE Phone IS NOT NULL
  AND (LENGTH(Phone) < 10 OR LENGTH(Phone) > 12);


--Final Profile Report
SELECT
    COUNT(*) AS Total,
    COUNT(DISTINCT Email) AS UniqueEmails,
    SUM(CASE WHEN City IS NULL THEN 1 ELSE 0 END) AS NullCities
FROM Customers;
