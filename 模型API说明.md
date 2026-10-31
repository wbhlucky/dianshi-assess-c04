# 模型 API 使用说明

HR 会单独发给你一个 API Key（形如 `sk-xxxxxxxx`）。这个 Key 仅供你的程序在运行时调用模型，不要写进代码或提交到仓库，用环境变量读取：

```bash
export LLM_API_KEY=sk-你的key
export LLM_BASE_URL=https://parvis.zjugpt.com/one-api/v1
```

（变量名随你定，上面只是示例；README 里写清楚即可。）

## 接口

OpenAI Chat Completions 兼容接口，任何 OpenAI SDK 都能直接用，Base URL 填 `https://parvis.zjugpt.com/one-api/v1`，客户端会自动拼接 `/chat/completions`，不要重复写 `/v1`。

| 用途 | 模型名 | 说明 |
|---|---|---|
| 文本（默认） | `deepseek-v4-flash` | 快、便宜，建议主力 |
| 文本（更强） | `deepseek-v4-pro` | 计费是 flash 的 3 倍，只在难的判定上用 |
| 看图（检测报告） | `qwen3.7-plus` | 支持 image_url 传图，可直接问图里的参数 |
| 纯 OCR | `qwen-vl-ocr` | 只输出图中文字，非常便宜 |

注意：deepseek 两个模型不支持传图；`qwen3-vl-flash` 虽在列表里但当前不可用，不要用。

连通性测试：

```bash
curl https://parvis.zjugpt.com/one-api/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"回复OK"}]}'
```

Python 示例：

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "回复OK"}],
)
print(resp.choices[0].message.content)
```

传图：把图片转成 base64，放在 `image_url` 里，格式 `data:image/jpeg;base64,...`，模型用 `qwen3.7-plus`。

## 额度

每个 Key 约 2000 万 token 的额度（按 deepseek-v4-flash 计；pro 按 3 倍扣）。正常做完 5 个训练包用不到十分之一。额度用完会返回 429 并提示 quota 不足，请联系 HR。

建议：先用小包 P1 调通流程再跑大包；对同一份文件的抽取结果做本地缓存，不要每次改代码都重新调模型。
