import unittest
import os
import openai

BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-35-turbo-0125")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-03-01-preview")


def chat(prompt):
    max_tokens = 4096  # range: [0, 4096]
    # if "deepseek" in MODEL_NAME.lower():
        
    if "gpt" in MODEL_NAME.lower():
        try:
            client = openai.AzureOpenAI(
                azure_endpoint=BASE_URL,
                api_version=API_VERSION,
                api_key=API_KEY,
            )
        except:
            client = openai.OpenAI(
                base_url=BASE_URL,
                api_key=API_KEY,
            )
    else:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # prompt = "上海天气怎么样？"
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        # extra_headers={"X-TT-LOGID": "${your_logid}"},  # header 参数传入。请务必带上 x-tt-logid，方便定位问题。logid 生成参考：https://bytedance.larkoffice.com/wiki/wikcnF5gKiIW655Tdqux88NMloh
    )
    return completion.choices[0].message.content


# print(chat('who are you'))
