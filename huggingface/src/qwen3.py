from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name="Qwen/Qwen3-235B-A22B"):
    """
    加载模型和分词器
    """
    try:
        logger.info(f"正在加载模型 {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("模型加载完成")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    生成回答
    """
    try:
        # 准备模型输入
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成文本
        logger.info("开始生成回答...")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # 解析输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析思考内容
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return thinking_content, content
    except Exception as e:
        logger.error(f"生成回答失败: {str(e)}")
        raise

def main():
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer()
        
        # 测试提示词
        prompt = "请用中文解释一下什么是大语言模型？"
        
        # 生成回答
        thinking_content, content = generate_response(model, tokenizer, prompt)
        
        # 打印结果
        print("\n思考过程：")
        print(thinking_content)
        print("\n最终回答：")
        print(content)
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
