# PPT 生成器 API 参考

## 主要函数

### generate_ppt(content, template, output_path)

生成PPT文件。

**参数:**
- `content` (str): Markdown 格式的内容
- `template` (str): 模板名称，如 "basic" 或 "professional"
- `output_path` (str): 输出文件路径

**返回:**
- `result` (dict): 包含生成状态和文件路径

### validate_content(content)

验证内容格式是否正确。

**返回:**
- `is_valid` (bool): 是否有效
- `errors` (list): 错误列表

## 内容格式

```markdown
# 标题页

## 第一页标题
- 要点1
- 要点2

## 第二页标题
内容详情...
```
