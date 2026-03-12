# SQL 性能优化指南

## 执行计划分析

使用 `EXPLAIN` 分析查询：

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 123;
```

### 关键指标
- **Seq Scan**: 全表扫描，通常需要优化
- **Index Scan**: 索引扫描，性能较好
- **Bitmap Heap Scan**: 位图扫描，适合多条件查询
- **Nested Loop**: 嵌套循环，小表驱动
- **Hash Join**: 哈希连接，适合大表

## 索引优化

### 创建索引
```sql
CREATE INDEX idx_customer_email ON customers(email);
CREATE INDEX idx_order_date_status ON orders(order_date, status);
```

### 索引原则
1. 高选择性列优先（区分度高的列）
2. 常用 WHERE 条件列
3. JOIN 关联列
4. ORDER BY / GROUP BY 列
5. 避免过多索引（影响写入性能）

### 复合索引顺序
```sql
-- 好的顺序：等值查询列在前，范围查询列在后
CREATE INDEX idx_good ON orders(customer_id, order_date);

-- 查询示例
SELECT * FROM orders
WHERE customer_id = 123
  AND order_date > '2024-01-01';
```

## 查询优化技巧

### 1. 避免 SELECT *
只查询需要的列，减少 I/O。

### 2. 使用 LIMIT
大数据集时使用分页。

```sql
SELECT * FROM logs
ORDER BY created_at DESC
LIMIT 100 OFFSET 0;
```

### 3. 避免在索引列上使用函数
```sql
-- 低效
WHERE DATE(created_at) = '2024-01-01'

-- 高效
WHERE created_at >= '2024-01-01'
  AND created_at < '2024-01-02'
```

### 4. 使用 EXISTS 替代 IN
```sql
-- 低效
SELECT * FROM customers
WHERE id IN (SELECT customer_id FROM orders);

-- 高效
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);
```

## 监控和诊断

### 慢查询日志
```sql
-- PostgreSQL
SELECT * FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### 表统计信息
```sql
-- 更新统计信息
ANALYZE table_name;

-- 查看表大小
SELECT pg_size_pretty(pg_total_relation_size('table_name'));
```
