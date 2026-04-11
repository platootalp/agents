# 基础 SQL 查询指南

## SELECT 语句基础

```sql
SELECT column1, column2
FROM table_name
WHERE condition
ORDER BY column1 DESC
LIMIT 10;
```

## JOIN 类型

### INNER JOIN
返回两个表中匹配的记录。

```sql
SELECT a.*, b.*
FROM table_a a
INNER JOIN table_b b ON a.id = b.a_id;
```

### LEFT JOIN
返回左表所有记录，右表匹配的记录。

```sql
SELECT customers.*, orders.order_id
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id;
```

## WHERE 子句

### 常用操作符
- `=` 等于
- `<>` 或 `!=` 不等于
- `>` 大于
- `<` 小于
- `BETWEEN` 范围
- `LIKE` 模糊匹配
- `IN` 列表匹配

### 示例
```sql
SELECT * FROM products
WHERE price BETWEEN 10 AND 100
  AND category IN ('electronics', 'books')
  AND name LIKE '%phone%';
```

## 聚合函数

- `COUNT(*)` - 计数
- `SUM(column)` - 求和
- `AVG(column)` - 平均值
- `MAX(column)` - 最大值
- `MIN(column)` - 最小值

```sql
SELECT
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
GROUP BY category
HAVING COUNT(*) > 5;
```
