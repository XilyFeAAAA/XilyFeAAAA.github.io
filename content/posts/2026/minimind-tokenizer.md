---
title: MiniMind 学习指北(二)：Tokenizer
date: 2026-01-22T13:47:19+08:00
featuredImage: http://img.xilyfe.top/img/20260122134824760.png
authors:
  - Xilyfe
series:
  - minimind
tags:
  - 大模型
  - 深度学习
lastmod: 2026-02-10T02:24:00+08:00
---
## Tokenizer

>在 CS336 的笔记中我已经完整介绍了一个 Tokenizer 是如何训练并且读取的，详情可见 [[cs336_assignment1]]。

简单来说，训练一个 tokenizer 经过以下步骤：
1. 通过正则分词，获得文本中全部 token，将其和 special_tokens 一起记录。
2. 不断把文本中出现频率最高的 token_pair 合并得到新 token，然后用新 token 替换文本中原先的 pair。
3. 重复上一步直到 vocab 达到指定规模。

上面的代码我们已经在 CS336 里实现过了，这一次我们通过 Huggingface 的 tokenizers 库直接生成。为了方便阅读，我先从如何得到一个 tokenizer 讲起。

首先 `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` 会根据参数自动加载对应的 tokenizer，假如我们传入的是模型名称，例如 "bert-base-uncased"、"gpt2"，它会先请求 config.json，从中判断 tokenizer 类型（BertTokenizer、GPT2Tokenizer、LlamaTokenizer 等），然后下载 tokenizer 必需的文件到本地。假如我们传入的是一个路径，它就会直接从文件夹中读取 tokenizer 的核心文件，它包括：

- tokenizer_config.json：里面包含 special_tokens、是否自动在文本开头添加 `bos_token` 等等 tokenizer 的配置信息。
- 词汇表文件（二选一）
	- vocab.json + merges.txt
	- tokenizer.json：实际上就是把 vocab.json + merges.txt 放到一起了

那我们训练 Tokenizer 就是要得到 tokenizer.json 和 tokenizer_config.json 两个文件。

```python
VOCAB_SIZE = 6400
SAVE_PATH = "out"
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def data_iterator(
    datapath: Union[str, PathLike[str]], max_sample: Optional[int] = None
):
    with open(datapath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sample is not None and i == max_sample:
                break
            data = json.loads(line)
            yield data["text"]


def train_tokenizer(datapath: Union[str, PathLike[str]]):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    iter = data_iterator(datapath)
    tokenizer.train_from_iterator(iter, trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(path.join(SAVE_PATH, "tokenizer.json"))
```

代码的核心就是 `tokenizer.train_from_iterator` 这个函数，我们提供一个数据集的迭代器还有一个训练器（例如这里我们用的是 BPE），就可以用 huggingface 提供的函数进行训练了。记得前面我们还说到，AutoTokenizer 除了 tokenizer.json 里面的词汇表和合并规则，还需要 tokenizer_config.json 的 tokenizer 配置信息，我们需要手动保存：

```python
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}",
}

with open(os.path.join(SAVE_PATH, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)
```

| 字段名                             | 解释                                                                        |
| ------------------------------- | ------------------------------------------------------------------------- |
| `add_bos_token`                 | 是否自动在文本开头添加 `bos_token`。                                                  |
| `add_eos_token`                 | 是否自动在文本末尾添加 `eos_token`。                                                  |
| `add_prefix_space`              | Byte-level 分词时是否在文本前加空格。通常英文中启用（True）更好，中文中设为 False。                      |
| `added_tokens_decoder`          | 特殊 token 的详细配置。包括 token 内容、是否为特殊 token、是否仅限单词等，key 是内部 token ID。          |
| `additional_special_tokens`     | 除了 `bos/eos/pad/unk` 外，额外声明的特殊 token 列表。当前为空。                             |
| `bos_token`                     | 起始 token，通常用于语言模型的开头控制符。                                                  |
| `clean_up_tokenization_spaces`  | 解码时是否清理 token 化带来的空格冗余。False 表示不清理。                                       |
| `eos_token`                     | 结束 token，通常用于语言模型输出结束的标记。                                                 |
| `legacy`                        | 设置为 `True` 兼容旧版本 `tokenizer` 行为。推荐保持默认。                                   |
| `model_max_length`              | 模型支持的最大 token 长度。超过将触发截断或报错。这里为 32768。                                    |
| `pad_token`                     | 用于对齐 padding 的特殊 token。                                                   |
| `sp_model_kwargs`               | SentencePiece 模型的额外配置参数（当前为 BPE，未使用，故为空）。                                 |
| `spaces_between_special_tokens` | 是否在特殊 token 之间自动添加空格，设置为 False。                                           |
| `tokenizer_class`               | 指定 tokenizer 类型。Hugging Face 使用 `"PreTrainedTokenizerFast"` 支持 Rust 实现加速。 |
| `unk_token`                     | 用于标记未知词（out-of-vocabulary）的 token。                                        |
| `chat_template`                 | Jinja2 模板字符串，用于格式化对话数据为模型输入格式。适用于 Chat 模型（如 LLaMA2-Chat、ChatGPT）。         |

测试 Tokenizer 很简单，我们对 prompt 进行编码之后再解码，将它和原始 prompt 对比，一致则说明成功了：

```python
def eval_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "你是一个优秀的聊天机器人，总是给我正确的回应！",
            },
            {"role": "user", "content": "你来自哪里？"},
            {"role": "assistant", "content": "我来自地球"},
        ],
        tokenize=False,
    )
    print(f"输入文本: \n{prompt}")
    print(f"词表长度 {len(tokenizer)}")
    input_ids = tokenizer(prompt)["input_ids"]
    print(input_ids)
    print(f"encoded prompt length = {len(input_ids)}")
    decoded_prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"decoded prompt == raw prompt: {decoded_prompt == prompt}")
```

>Chat 模型并不是我们发送什么他就输入什么，它会把输入内容改写为一种带特殊标记的结构化格式（就是 tokenizer_config.json 里面 Jinja2 字符串模板）。所以真正的 input 应该写成 `tokenizer.apply_chat_template(prompt)`。