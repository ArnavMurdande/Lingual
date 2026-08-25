CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_name VARCHAR(100),
    region VARCHAR(50),
    segment VARCHAR(50)
);

CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    sub_category VARCHAR(50)
);

CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE,
    day INT,
    month INT,
    quarter INT,
    year INT
);

CREATE TABLE fact_sales (
    sales_key INT PRIMARY KEY,
    date_key INT,
    customer_key INT,
    product_key INT,
    quantity_sold INT,
    sales_amount DECIMAL(10,2),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key)
);

INSERT INTO dim_customer VALUES
(1, 'Alice Corp', 'North', 'Enterprise'),
(2, 'Beta LLC', 'South', 'SMB');

INSERT INTO dim_product VALUES
(1, 'Laptop Pro', 'Electronics', 'Computers'),
(2, 'Office Chair', 'Furniture', 'Chairs');

INSERT INTO dim_date VALUES
(20250101, '2025-01-01', 1, 1, 1, 2025),
(20250102, '2025-01-02', 2, 1, 1, 2025);

INSERT INTO fact_sales VALUES
(1, 20250101, 1, 1, 10, 15000.00),
(2, 20250102, 2, 2, 5, 1000.00);

SELECT SUM(sales_amount) AS total_sales
FROM fact_sales;

SELECT c.customer_name, SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_customer c
ON f.customer_key = c.customer_key
GROUP BY c.customer_name;

SELECT p.category, SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_product p
ON f.product_key = p.product_key
GROUP BY p.category;

SELECT d.full_date, SUM(f.sales_amount) AS daily_sales
FROM fact_sales f
JOIN dim_date d
ON f.date_key = d.date_key
GROUP BY d.full_date
ORDER BY d.full_date;
