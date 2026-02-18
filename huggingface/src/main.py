from torch import nn
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import logging
import time
from huggingface_hub import HfFolder
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_with_retry(model_name, max_retries=3, retry_delay=5):
    """带重试机制的模型加载函数"""
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试加载模型 {model_name} (第 {attempt + 1} 次尝试)")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            logger.warning(f"加载模型失败: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                raise

def sentiment_analysis():
    try:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        logger.info("正在初始化情感分析模型...")
        model, tokenizer = load_model_with_retry(model_name)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        # 测试文本
        test_texts = [
            "I love this movie!",
            "This is terrible.",
            "I'm not sure how I feel about this.",
            "Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."
        ]
        
        logger.info("开始情感分析...")
        results = classifier(test_texts)
        
        # 打印结果
        for text, result in zip(test_texts, results):
            print(f"文本: {text}")
            print(f"情感: {result['label']}")
            print(f"置信度: {result['score']:.4f}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        raise

# def tokenizer_test():
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     print(tokenizer("We are very happy to show you the 🤗 Transformers library."))
#     print(tokenizer.tokenize("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."))

def scaled_dot_product_attention():
    try:
        model_ckpt = "bert-base-uncased"
        logger.info(f"正在加载模型 {model_ckpt}...")
        model, tokenizer = load_model_with_retry(model_ckpt)

        text = "time flies like an arrow"
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        print(inputs.input_ids)

        config = AutoConfig.from_pretrained(model_ckpt)
        token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        print(token_emb)

        inputs_embeds = token_emb(inputs.input_ids)
        print(inputs_embeds.size())
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置环境变量，使用本地缓存
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # 设置模型缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # sentiment_analysis()
    # tokenizer_test()
    scaled_dot_product_attention()