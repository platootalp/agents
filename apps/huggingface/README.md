# Hugging Face 情感分析项目

这是一个使用Hugging Face Transformers库进行情感分析的示例项目。

## 项目说明

本项目使用Hugging Face的Transformers库，通过预训练模型对文本进行情感分析。目前使用的是`distilbert-base-uncased-finetuned-sst-2-english`模型，这是一个在SST-2数据集上微调的英语情感分析模型。

## 环境要求

- Python >= 3.11
- uv (Python包管理器)
- 依赖包：
  - transformers
  - torch
  - numpy<2.0.0 (由于当前兼容性问题，需要降级到1.x版本)

## 安装步骤

1. 使用uv安装依赖：
```bash
uv pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python src/main.py
```

## 注意事项

1. 首次运行时会自动下载预训练模型，请确保网络连接正常
2. 如果遇到模型下载超时，可以尝试：
   - 使用代理
   - 手动下载模型并放置在本地缓存目录
   - 使用国内镜像源

## 常见问题

1. NumPy版本兼容性问题：
   - 解决方案：使用uv安装正确版本的NumPy
   ```bash
   uv pip install "numpy<2.0.0"
   ```

2. 模型下载超时：
   - 解决方案：设置环境变量使用国内镜像
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 项目结构

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
└── src/
    └── main.py
```

## 示例输出

运行程序后，您将看到类似以下的输出：

```
INFO:__main__:正在初始化情感分析模型...
INFO:__main__:开始情感分析...
文本: I love this movie!
情感: POSITIVE
置信度: 0.9999
--------------------------------------------------
文本: This is terrible.
情感: NEGATIVE
置信度: 0.9996
--------------------------------------------------
文本: I'm not sure how I feel about this.
情感: NEGATIVE
置信度: 0.9992
--------------------------------------------------
``` 