---
title: LLM 清洗数据
date: 2026-03-29T12:35:46+08:00
featuredImage: http://img.xilyfe.top/img/20260331232449417.png
authors:
  - Xilyfe
series:
  - LLM
tags:
  - 大模型
  - 数据集
lastmod: 2026-04-03T08:15:49+08:00
---
{{< admonition type=info title="Summary">}} 
从互联网获取的原始数据就像未经加工的矿石，其中真正有价值的"精矿"可能只占很小的比例。本章将深入探讨预训练数据清洗的三大核心技术：启发式过滤规则用于剔除低质量文档，大规模去重技术用于消除重复内容，隐私数据清洗用于保护用户信息。掌握这些技术后，读者将能够构建工业级的数据清洗流水线，将原始网页数据转化为高质量的预训练语料。
{{< /admonition >}}

## 1. 启发式过滤

启发式过滤是数据清洗的第一步，它是一系列的 rule-based filter。通过这些固定的规则，可以快速筛选出明显低质量的文档。虽然这些规则看起来简单，但在实践中能够过滤掉大部分噪声数据，是性价比极高的清洗手段。

### 1.1 规则过滤

除了语言识别和困惑度过滤，还有一系列简单但有效的启发式规则，可以快速剔除明显的低质量内容。这些规则的设计来源于对大量数据的观察和经验总结：
1. **长度过滤**是最基本的规则。过短的文档没有训练价值，应该直接移除。过长的文档可能需要截断或分段处理。典型的阈值设定是：最小长度 200 字符或 50 词，最大长度 100,000 字符。
2. **特殊字符比例**可以识别出大量噪声内容。如果一个文档中非字母数字字符的比例过高，很可能是代码残留、乱码或格式错误。类似地，数字比例过高可能表示是日志文件或数据表格。
3. **词汇多样性**衡量文档的信息丰富程度。一个只使用 10 个不同词汇的文档显然不如使用 500 个不同词汇的文档有价值。常用的指标是 Type-Token Ratio（TTR），即唯一词数与总词数的比值。

```python
class RuleFilter:
    def __init__(
        self,
        min_length: int = 0,
        max_length: float = float("inf"),
        max_special_ratio: float = 1,
        max_digit_ratio: float = 1,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_special_ratio = max_special_ratio
        self.max_digit_ratio = max_digit_ratio

    def _check_length(self, text: str) -> bool:
        length = len(text)

        return self.min_length <= length <= self.max_length

    def _check_special_char(self, text: str) -> bool:
        special = len(re.findall(r"[^\w\s]", text, re.UNICODE))
        ratio = special / len(text)
        return ratio <= self.max_special_ratio

    def _check_digit(self, text: str) -> bool:
        digits = len(re.findall(r"\d", text))
        ratio = digits / len(text)
        return ratio <= self.max_digit_ratio

    def _check(self, text: str) -> bool:
        return (
            self._check_length(text)
            and self._check_special_ratio(text)
            and self._check_digit_ratio(text)
        )

    def filter(self, texts: list[str]) -> list[str]:
        res = []
        for text in tqdm(texts, desc="Rule Filtering"):
            if self._check(text):
                res.append(text)
        return res

```

### 1.2 语言过滤

对于训练中文模型而言，首先需要从海量数据中筛选出中文内容，这就需要准确的语言识别能力。**FastText** 是目前最常用的工具。它由 Facebook AI Research 开发，预训练模型支持 176 种语言的识别，速度极快，准确率也相当高。FastText 提供两个预训练模型：`lid.176.bin` 是完整版本，准确率更高但体积较大；`lid.176.ftz` 是压缩版本，体积小但准确率略低。对于大规模数据处理，建议使用完整版本。

```python
class LanguageFilter:
    def __init__(
        self,
        model_path: str = "lib.176.bin",
        min_len: int = 10,
        truncate_len: int = 1024,
        tgt_lan: str = "zh",
        min_conf: float = 0.7,
    ):
        self.model = fasttext.load_model(model_path)
        self.tgt_lan = tgt_lan
        self.min_len = min_len
        self.min_conf = min_conf
        self.truncate_len = truncate_len

    def detect_lan(self, text: str) -> tuple[bool, float]:
        text = text.replace("\n", " ")[: self.truncate_len]

        preds = self.model.predict(text, k=1)
        lang = preds[0][0].replace("__label__", "")
        conf = preds[1][0]

        return lang if conf >= self.min_conf else None, conf

    def filter(self, texts: list[str]) -> tuple[list, list]:
        results = []
        confidence = []

        for text in tqdm(texts, desc="Language Filtering"):
            lang, conf = self.detect_lan(text)
            if lang == self.tgt_lan:
                confidence.append(conf)
                results.append(text)
        return results, confidence
```

### 1.3 质量过滤

困惑度的原理是：

$$
\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, ..., w_{i-1})\right)
$$

它可以衡量模型对一个句子的意外程度，如果 PPL 数值越大说明模型对这句话越意外，一般来说这句话就越没有逻辑越混乱。同时计算长文本 PPL 的时候，我们通常用滑动窗口来避免 token 超出上下文长度的问题。LLM 计算 PPL 时，每个 token 的概率依赖前面的上下文，但模型的上下文窗口是有限的。假如不用滑动窗口，我们采用朴素分块，每块的第 1 个 token 对模型来说是凭空出现，缺乏上文信息，模型会给它极低的概率。这些 token 的 PPL 会严重拖低整体均值，如图所示：

![image.png](http://img.xilyfe.top/img/20260401192956303.png)


更合理的做法是使用滑动窗口策略来评估固定长度模型的困惑度，以便模型在进行每个预测时具有更多上下文。每个 token 被预测时，它前面最多有 $W - 1$ 个 token 作为 context，大幅降低冷启动问题。代价是需要更多次前向传播，stride 越小，PPL 越准，但计算量越大：

![image.png](http://img.xilyfe.top/img/20260401105243627.png)

```python
class PerplexityFilter:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        batch_size: int = 8,
        stride: int = 512,
        max_length: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.batch_size = batch_size
        self.stride = stride
        self.max_length = max_length
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()

    def filter(self, texts: list[str]) -> list[str]:
        all_ppls = []
        n_texts = len(texts)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        for i in tqdm(range(0, n_texts, self.batch_size), desc="PPL Filtering"):
            inputs = self.tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                return_tensors="pt",
                truncation=False,
            )

            bs, seq_len = inputs.input_ids.size()

            prev_end_loc = 0
            nll_sum = torch.zeros(bs, device=self.device)
            token_count = torch.zeros(bs, device=self.device)

            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.stride, seq_len)
                chunk_ids = inputs.input_ids[:, begin_loc:end_loc].to(self.device)
                chunk_mask = inputs.attention_mask[:, begin_loc:end_loc].to(self.device)

                with torch.no_grad():
                    logits = self.model(chunk_ids, chunk_mask).logits

                shift_logits = logits[:, :-1, :]
                shift_labels = chunk_ids[:, 1:]
                shift_mask = chunk_mask[:, 1:]

                loss = loss_fn(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
                loss = loss.view(bs, -1)
                loss = loss * shift_mask

                pred_len = end_loc - prev_end_loc
                loss = loss[:, -pred_len:]
                mask = shift_mask[:, -pred_len:]

                nll_sum += loss.sum(dim=1)
                token_count += mask.sum(dim=1)

                prev_end_loc = end_loc
                if end_loc >= seq_len:
                    break

            ppl = torch.exp(nll_sum / token_count)
            all_ppls.extend(ppl.detach().cpu().tolist())

        return all_ppls
```

## 2. 数据去重

### 2.1 精确去重

精确去重基于完全一致的匹配原则，即对每一个数据样本计算一个确定性的标识符，并通过比较标识符是否相同来判断样本是否完全一致。对于具有相同标识的样本仅保留其中一个，其余样本被移除。该方法实现简单、计算效率高，能够有效消除完全重复的数据样本，但无法识别语义相同或高度相似的重复内容例如轻微改写、格式变化或局部修改的文本。

```python
def exact_dedup(texts: list[str]) -> list[str]:
    seen = set()
    results = []

    for text in texts:
        hash = hashlib.sha256(text.encode("utf-8")).digest()
        if hash not in seen:
            seen.add(hash)
            results.append(text)

    return results
```

你可能会有这样的疑惑，既然我们在 `results` 里面存了返回的结果，那我们用 `seen` 存历史哈希值有什么用呢？在现在的场景下我们确实可以把 text 存在 set 里面，这样 python 会自动用哈希值判断文本是否重复了。但是在大数据量情况下，我们不一定能把所有的文本都存入内存，所以计算出哈希值存在 set 里面可以大幅度减少存储占用。

### 2.2 模糊去重

#### 2.2.1 SimHash

SimHash 的原理是，它会把文档切分成一系列特征，然后统计数据的分布，计算各个特征的加权哈希值最后得到这个数据的 **指纹**。如果两个文档越相似 → 它们的 SimHash 指纹汉明距离越小。具体用一个例子看看他是怎么计算的：

1. 对文档进行分词，用 jieba 或者 BPE tokenizer 都可以。例如 `words = ['不能','复现','的','软件','不算','开源软件']`
2. 计算每个词的哈希值，我们可以得到：
	1. 不能:38f9286be23a182e7403ef05db293b49
	2. 复现:90c33da6cdc6c148d6eab60f7f4926e1
	3. 的:01d7aa494b0727f8db77be1d3685de9e
	4. 软件:f0dfc65b71f0f4c075027ecbfe66ef7c
	5. 不算:7c85dceaa0f0e8bb3a351a059ed05d04
	6. 开源软件:ed507ba1538b7c4098b3f82e7ba8af9c
3. 然后把他转为对应的二进制数，例如：不能 → 0110111010100101...
4. 然后 1 保持不变 0 变成 -1，不能 → \[-1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1]
5. 进行向量累加。对每个特征（假设该特征在句子里出现 w 次）做如下操作：
	1. 如果该位是 1 → $+w$
	2. 如果该位是 0 → $-w$
6. 通过前面五个步骤，我们得到的向量可能形如 \[3,5,-1,-5,2,-6,7,...]。然后我们进行二值分类，大等于零就令其为 1，小于零就位 0，最终得到 SimHash 指纹 101101001010...。
7. 对于两个 SimHash 指纹，我们通过海明距离来判断它们的相似度。例如 010111 和 010001 有一位不同，所以它们的海明距离为 1。一般来说海明距离 ≤ 3~8 都认定为两个文档相似。

从上述流程中可以看出来，由于 SimHash 的每一个 bit 都是由不同的特征加权得到的，所以最终少许的差异不会影响整体判断，并且 SimHash 长度越长，他的判断越准确。

```python
class SimHash:
    def __init__(self, hash_bits: int):
        self.bits = hash_bits

        self.texts: list = []
        self.token_cache: dict = {}
        self.idf: dict = {}

    def tokenize(self, text: str) -> list[str]:
        if text in self.token_cache:
            return self.token_cache[text]

        cleaned = PUNCT_RE.sub(" ", text)
        tokens = jieba.lcut(cleaned)

        self.token_cache[text] = tokens
        return tokens

    def _token_hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode()).hexdigest()
        return int(digest, 16) & ((1 << self.bits) - 1)

    def build(self, texts: list[str]):
        self.texts = texts

        N = len(self.texts)
        df = Counter()

        for text in tqdm(self.texts, docs="Calculating IDF"):
            for token in set(self.tokenize(text)):
                df[token] += 1

        idf: dict[str, float] = {}
        for token, count in df.items():
            idf[token] = math.log(N / (1 + count))

        return idf

    def calculate(self, text: str) -> int:
        tokens = self.tokenize(text)
        num_tokens = len(tokens)
        tf = Counter(tokens)
        v = [0.0] * self.bits

        for token, count in tf:
            tf = count / num_tokens
            weight = tf * self.idf.get(token)

            h = self._token_hash(token)
            for i in range(self.bits):
                if h >> i & 1:
                    v[i] += weight
                else:
                    v[i] -= weight

        fingerprint = 0
        for k in range(self.bits):
            if v[k] >= 0:
                fingerprint |= 1 << k

        return fingerprint

    def hamming_distance(self, a: int, b: int) -> int:
        return bin((a ^ b) & ((1 << self.bits) - 1)).count("1")

    def similarity(self, a: int, b: int) -> float:
        return 1.0 - self.hamming_distance(a, b, self.bits) / self.bits
```

一开始学完 SimHash 我有个问题：我们为每一个文本都计算出了它的 SimHash 值，那我们需要两两对比一共 $O(n^2)$ 的时间复杂度吗？这里我们先忽略这一点，讲一下 MinHash。

#### 2.2.2 MinHash

MinHash 需要从衡量两个文档的相似度说起，Jaccard 相似度用于描述两个集合的相似程度，假设有两个集合 A 和 B ，两个集合的相似度为交集的元素个数除以并集的元素个数，公式为：

$$
J \left(\right. A , B \left.\right) = \frac{\left|\right. A \cap B \left|\right.}{\left|\right. A \cup B \left|\right.}
$$

但是对海量文本直接求 Jaccard 相似度复杂度太高，两个文档需要逐个词比较。为降低复杂度，我们使用两个文档的最小哈希值相等的概率来等价于两个文档的 Jaccard 相似度，并可以证明两者是相等的。假设我们有数据集：$S_1=\{a,b,d\}$ $S_2=\{b,c,d\}$，我们可以把集合变成一个矩阵：

| 行号  | 元素  | S1  | S2  |
| --- | --- | --- | --- |
| 0   | a   | 1   | 0   |
| 1   | b   | 1   | 1   |
| 2   | c   | 0   | 1   |
| 3   | d   | 1   | 1   |
| 4   | e   | 0   | 0   |

然后 MinHash 算法会随机打乱行，取每列第一个 1。具体来说，会选一个哈希函数，把行号重新映射对每个集合，从上往下扫描，第一个值为 **1** 的行，就是这个集合的 MinHash 值。例如：

| 原行号 | 打乱后行号 | 元素  | S1  | S2    |
| --- | ----- | --- | --- | ----- |
| 2   | 0     | c   | 0   | 1     |
| 0   | 1     | a   | 1   | 0     |
| 4   | 2     | e   | 0   | 0     |
| 3   | 3     | d   | 1   | **1** |
| 1   | 4     | b   | 1   | 1     |

可以看到，打乱后 S1 出现的第一个 1 是元素 a，S2 打乱后出现的第一个 1 是元素 c，所以 MinHash(S1)=a，MinHash(S2)=c，它们的 MinHash 值不同。但单次 MinHash 只给一个 0/1 的比较结果，方差很大。用 k 个不同的哈希函数，各自打乱一次，得到 k 个 MinHash 值。k 越大，估计越准，但计算量也越大。

| 哈希函数           | MinHash(S1) | MinHash(S2) | 相等？    |
| -------------- | ----------- | ----------- | ------ |
| h₁             | b           | b           | ✓      |
| h₂             | a           | c           | ✗      |
| h₃             | d           | d           | ✓      |
| h₄             | b           | b           | ✓      |
| h₅             | a           | d           | ✗      |
| h₆             | d           | c           | ✗      |
| h₇             | b           | b           | ✓      |
| h₈             | d           | d           | ✓      |
| **估计 Jaccard** | 5 个相同 / 8 个 |             | ≈ 0.63 |

{{< admonition type=question title="为什么概率=Jaccard相似度？">}} 
对任意两个集合 S1、S2，每一行只有 3 种情况：
1. X：两列都是 1（交集元素）
2. Y：只有一列是 1（差集元素）
3. Z：两列都是 0（无关）

随机打乱后，从上往下扫描，只要还没碰到 X 或 Y，就一直是 Z，跳过。所以真正决定结果的，是第一个非 Z 行是 X 还是 Y。设有 **x** 行是 X，**y** 行是 Y：  
• 第一个非Z行是 X 的概率 = **x/(x+y)** → 此时 h(S1)=h(S2)  
• 第一个非Z行是 Y 的概率 = **y/(x+y)** → 此时 h(S1)≠h(S2)

$$
P(h(S_1)=h(S_2)) = \frac{x}{x+y}=\frac{\text{并集}}{\text{交集}}=\text{Jaccard}(S_1,S_2)
$$

{{< /admonition >}}

至此为止，我们说的都是 **理论上的 MinHash**。实际上我们不可能用一个超大的矩阵存下所有文档的词汇表，这个超大的矩阵可能稀疏到 95% 以上都是 0。具体来说：
- 我们把每个 token 看作矩阵的“行号”。
- 用一个哈希函数 h 把 token 映射到一个很大的随机整数空间（例如 64 位或 128 位）。
- 因为 h 是随机的、均匀的、独立的，它等价于给所有 token 随机排了一个顺序（即一次随机置换）。
- 因此，对一个集合 S 的 MinHash 值就等于（矩阵列 S 的第一个 1 就是最小的哈希值）：

$$
\text{MinHash}_h(S) = \min_{x \in S} h(x)
$$

```python
def minhash_signature(doc_tokens, num_perm=128):
    signature = []
    for i in range(num_perm):
        # h_i 是第 i 个独立的哈希函数（通常用 MurmurHash + 不同 seed）
        min_val = float('inf')
        for token in doc_tokens:           # ← 这里就是“对每个 token 进行 hash”
            val = hash_i(token, seed=i)    # hash(token)
            if val < min_val:
                min_val = val
        signature.append(min_val)
    return signature
```

这个代码应该很简洁了，一共计算 `num_perm` 个 MinHash 值，每次都对所有 token 计算他们的最小的哈希值，也就对应之前的 **每列第一个 1**。但是这个代码时间复杂度比较高，我们可以调换两个 for 循环的顺序，这样只需要计算 $O(\text{tokens})$ 次的哈希值：

```python
_MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1，大质数
_MAX_HASH = (1 << 32) - 1  # 32-bit 最大值


class MinHash:
    def __init__(self, num_perm: int = 128) -> None:
        self.num_perm = num_perm
        self.signature = [_MAX_HASH] * num_perm
        self.params = [
            (
                random.randint(1, _MERSENNE_PRIME - 1),
                random.randint(0, _MERSENNE_PRIME - 1),
            )
            for _ in range(num_perm)
        ]

    def hash(self, value: str) -> int:
        return int(hashlib.md5(value.encode()).hexdigest(), 16) & _MAX_HASH

    def update(self, features: set[str]) -> "MinHash":
        for feat in features:
            _h = self.hash(feat)
            for i, (a, b) in enumerate(self.params):
                hashed = (a * _h + b) % _MERSENNE_PRIME
                self.signature[i] = min(self.signature[i], hashed)
        return self

    def jaccard(self, other: "MinHash") -> float:
        assert self.num_perm == other.num_perm
        matches = sum(a == b for a, b in zip(self.signature, other.signature))
        return matches / self.num_perm

```

#### 2.2.3 局部敏感哈希

到这里我们先总结一下 SimHash 和 MinHash 的区别：
1. MinHash 判断的是 **两个文本有多少内容是一样的？**
2. SimHash 判断的是 **两个文本的词袋向量夹角是不是差不多？**

因为 MinHash 的本质是 Jaccard 相似度，也就是 $J \left(\right. A , B \left.\right) = \frac{\left|\right. A \cap B \left|\right.}{\left|\right. A \cup B \left|\right.}$，它只看有没有这个词而不看语义。SimHash 的本质是每个词计算哈希值变成向量，然后进行按权重累加，最后二值化得到 n 位比特。它在做 **向量投影**，整体方向接近那么哈希值就接近。

---

之前我们提到一个问题，SimHash 和 MinHash 都只能两两计算文本的相似度，那么时间复杂度也太高了吧？在具体实践中会用 LSH 局部敏感哈希的方法，来避免所有人互相比较的 $O(n^2)$ 复杂度。

局部敏感哈希的原理是：LSH 将SimHash 和 MinHash 的哈希值进行分块，如果两个文本的 signature 在某个块完全相同，那我们就认为这两个文本是**候选的**相似文本，注意不是肯定相似而是可能相似。为什么我们可能这么认为呢？假设两篇文档的相似度为 $s$，那么它们**任意一个哈希值相同的概率**也是 $s$。现在，我们来计算两个文档被判定为候选对的概率：
1. 一个块内的所有 r 个哈希值都相同的概率： $s^r$。
2. 一个块内至少有一个哈希值不同的概率： $1 - s^r$。
3. 所有 b 个块都至少有一个哈希值不同的概率： $(1 - s^r)^b$。
4. 因此至少有一个带完全相同的概率： $P = 1 - (1 - s^r)^b$。

这个函数 $P(s)$ 是一个神奇的S形曲线：
- 对于高相似度的文档对（比如 $s$ 接近1）： $s^r$ 接近 1，所以 $(1 - s^r)^b$ 会迅速趋近于 0，结果 $P$ 非常接近 1，这意味着几乎一定能被选中。
- 对于低相似度的文档对（比如 $s$ 接近0）： $s^r$ 接近 0，所以 $1 - s^r$ 接近1，$(1 - s^r)^b$ 也接近 1，结果 $P$ 非常接近0，这意味着几乎一定不会被选中。

所以我们控制 $r$ 和 $b$ 的大小就可以控制这个 $S$ 曲线的陡峭程度和阈值。比如，你想让相似度超过80%的文档对几乎必定成为候选，就可以计算出对应的 $r$ 和 $b$。MinHash 的 `num_perm` 和 SimHash 的 `num_bits` 越大，那么识别的准确率高越高误差越小。$r$ 不变的情况下，$b$ 越小相似度阈值越大。

MinHashLSH 代码如下：

```python
class MinHashLSH:
    def __init__(
        self,
        num_band: int = 20,
        num_perm: int = 128,
        use_split: bool = False,
        use_ngram: bool = False,
    ) -> None:
        assert use_split != use_ngram

        self.num_band = num_band
        self.num_perm = num_perm
        self.num_row = num_perm // num_band
        self.use_split = use_split
        self.use_ngram = use_ngram

        self.buckets = [defaultdict(set) for _ in range(num_band)]
        self.params = [
            (
                random.randint(1, _MERSENNE_PRIME - 1),
                random.randint(0, _MERSENNE_PRIME - 1),
            )
            for _ in range(num_perm)
        ]

    @staticmethod
    def _split(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _ngram(text: str, ngram: int = 3) -> list[str]:
        return [text[i : i + ngram] for i in range(len(text) - ngram + 1)]

    def _get_minhash(self, text: str) -> MinHash:
        split_fn = self._split if self.use_split else self._ngram
        features = split_fn(text)

        minhash = MinHash(self.num_perm, self.params)
        minhash.update(features)

        return minhash

    def add(self, doc_id: int, text: str):
        minhash = self._get_minhash(text)
        signature = minhash.signature

        for i in range(self.num_band):
            band_vals = signature[i * self.num_row : (i + 1) * self.num_row]
            band_hash = hash(tuple(band_vals))
            self.buckets[i][band_hash].add(doc_id)

    def query(self, text: str):
        signature = self._get_minhash(text).signature
        candidate = set()

        for i in range(self.num_band):
            band_vals = signature[i * self.num_row : (i + 1) * self.num_row]
            band_hash = hash(tuple(band_vals))

            if band_hash in self.buckets[i]:
                candidate.update(self.buckets[i][band_hash])

        return candidate
```

SimHash 代码如下：

```python
class SimHashLSH:
    def __init__(
        self,
        num_band: int = 16,
        hash_bits: int = 128,
    ) -> None:

        self.num_band = num_band
        self.hash_bits = hash_bits
        self.rows_per_band = hash_bits // num_band

        self.id2fp = {}
        self.simhash = SimHash(hash_bits)
        self.buckets = [defaultdict(set) for _ in range(num_band)]

    def fit(self, texts: list[tuple]) -> None:
        self.simhash.build([text for _, text in texts])

        for tid, text in texts:
            self.add_single(tid, text)

    def add_single(self, tid: str, text: str) -> None:
        fp = self.simhash.compute(text)
        self.id2fp[tid] = fp

        for band_idx in range(self.num_band):
            start_bit = band_idx * self.rows_per_band
            band_mask = (1 << self.rows_per_band) - 1
            band_value = (fp >> start_bit) & band_mask
            self.buckets[band_idx][band_value].add(tid)

    def query(self, text: str) -> set[str]:
        """查询与给定文本潜在相似的文档 id"""
        fp = self.simhash.compute(text)
        candidates = set()

        for band_idx in range(self.num_band):
            start_bit = band_idx * self.rows_per_band
            band_mask = (1 << self.rows_per_band) - 1
            band_value = (fp >> start_bit) & band_mask

            if band_value in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_value])

        return candidates
```

接下来我们用一个 demo 看看 SimHashLSH 和 MinHashLSH 的区别：

```
【查询文本】: The quick brown fox jumps over the lazy dog

【MinHashLSH 筛选出的候选 ID】: {'doc_similar', 'doc_partial', 'doc_exact'}

【MinHashLSH精筛结果】:
ID: doc_similar  | 相似度: 0.8750 | 内容: The quick brown fox jumps over the dog
ID: doc_partial  | 相似度: 0.6000 | 内容: A quick brown fox leaps over a lazy dog
ID: doc_exact    | 相似度: 1.0000 | 内容: The quick brown fox jumps over the lazy dog

【SimHashLSH 筛选出的候选 ID】: {'doc_similar', 'doc_partial', 'doc_diff', 'doc_exact', 'doc_ai'}

【SimHashLSH精筛结果】:
ID: doc_similar  | 相似度: 0.9297 | 内容: The quick brown fox jumps over the dog
ID: doc_partial  | 相似度: 0.7188 | 内容: A quick brown fox leaps over a lazy dog
ID: doc_diff     | 相似度: 0.6953 | 内容: Python is a high-level programming language
ID: doc_exact    | 相似度: 1.0000 | 内容: The quick brown fox jumps over the lazy dog
ID: doc_ai       | 相似度: 0.5938 | 内容: Large language models are trained on massive data
```

可以看出来，MinHashLSH 计算出来的结果更加准确，doc_diff 和 doc_ai 都给出了非常低的相似度，没有被召回。反之 SimHashLSH 给完全不相似的这两个句子给出了 0.69 和 0.59 的分数，**证明了 SimHash 在低相似度区域的不可靠性**。不过这个例子中 SimHashLSH 假阳性高，可能也是因为样本太少导致的。

{{< admonition type=question title="SimHashLSH 给改写的句子分数更高，是不是说明能力更强呢？">}} 
看到结果我在想，SimHashLSH 给改写但是语义相同的句子更高的相似度，是不是说明 SimHash 能识别语义相似的改写？这其实是个误区，出现这种情况只是因为 SimHash 本质是**带权重的随机投影**，权重 = TF-IDF。在这个小语料中，`jumps` 和 `leaps` 都只出现在一篇文档中，它们的 IDF 都较高，但因为 `jumps` 在原句中出现一次，`leaps` 在 doc_partial 中出现一次，TF 归一化后权重相近。

SimHash 的优势在于 **极快的计算速度** 和 **极低的内存占用**，它适合做适合海量数据的近似重复检测。它的缺点也很明显：
1. 对中等相似度的文档判别能力差：此时汉明距离处于中间区域，LSH 的命中概率不稳定，假阳性和假阴性都会增加。
2. 对短文本效果差：词太少时，SimHash 向量稀疏，指纹容易受一两个词的权重支配，失去稳定性。

所以在实际操作中一般会结合 SimHash 和 MinHash，先用 SimHash 粗筛再用 MinHash 精排。对 SimHash 我们筛选去掉海明距离小于 3 的句子，因为高相似度的文档 SimHash 的命中概率比较高，不容易误判。然后对剩下的候选文档计算 MinHash 相似度，得到更准确的相似度，根据阈值决定是否重复。
{{< /admonition >}}

## 3. 隐私数据清洗

训练数据中不可避免地包含个人身份信息（Personally Identifiable Information, PII），PII 可以分为直接标识符和准标识符两类。直接标识符可以单独识别个人身份，如姓名、身份证号、社会保障号、电话号码、电子邮箱。准标识符单独难以识别个人，但组合使用可能导致识别，如出生日期、邮政编码、职业、工作单位。在训练数据中保留 PII 存在多重风险。首先是隐私泄露风险：模型可能"记住"训练数据中的敏感信息，在推理时被恶意提取。其次是合规风险：违反数据保护法规可能导致巨额罚款。最后是声誉风险：如果模型输出他人隐私信息，会严重损害企业形象。

![image.png](http://img.xilyfe.top/img/20260403183144118.png)

### Microsoft Presidio

Presidio 是微软开源的 PII 识别和匿名化工具包，支持多种语言和多种 PII 类型。它采用模块化设计，包含两个核心组件：Analyzer 负责在文本中识别 PII 实体，Anonymizer 负责对识别出的 PII 进行处理（如替换、掩码、删除）。

```python
from typing import Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PII:
    def __init__(self, entities: list[str], threshold: Optional[float] = None):
        self._entities = entities
        self._threshold = threshold

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "DATE_TIME": OperatorConfig("keep", {}),  # 日期时间通常可以保留
        }

    def analyze(self, text: str, lan: str = "en") -> list:
        return self._analyzer.analyze(
            text=text,
            language=lan,
            entities=self._entities,
            score_threshold=self._threshold,
        )

    def anoymize(self, text: str, analyzer_results: list):
        return self._anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results, operators=self._operators
        ).text
```

简单写个 Demo 测试一下：

```
原始文本：
My name is John Doe, my phone number is 555-123-4567, and my email is john.doe@example.com. My credit card is 4111-1111-1111-1111.

识别到的实体：
  - 类型: PERSON, 位置: 11-19, 置信度: 0.85
  - 类型: PHONE_NUMBER, 位置: 46-58, 置信度: 0.75
  - 类型: EMAIL_ADDRESS, 位置: 73-91, 置信度: 0.80
  - 类型: CREDIT_CARD, 位置: 108-131, 置信度: 0.60

匿名化后的文本：
My name is <PERSON>, my phone number is <PHONE_NUMBER>, and my email is <EMAIL_ADDRESS>. My credit card is <CREDIT_CARD>.
```

### 中文 PII

Presidio 的问题是他对中文数据的处理不够好，所以中文数据通常需要补充基于正则表达式的规则匹配。
- 电话号码：`r'1[3-9]\d{9}'`
- 邮件地址：`r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'`
- IP 地址：`r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'`
- 等等

## 4. 基准测试集防污染

TODO

## 5. 基于模型评分

TODO

## 6. 完整 pipeline


