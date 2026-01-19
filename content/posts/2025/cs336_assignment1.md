---
title: "Assignment 1: Tokenizer & Transformer"
date: '2025-12-31T16:51:11+08:00'
authors: [Xilyfe]
series: ["CS336"]
tags: ["大模型"]
--- 

# BPE Train

## Problems

**1. Understanding Unicode**

(a) What Unicode character does chr(0) return?一个空字符。
(b) How does this character’s string representation (__repr__()) differ from its printed representa-  
tion? __repr__() 输出的是它的字节表示，print 输出的是空字符。
(c) What happens when this character occurs in text? It may be helpful to play around with the  
following in your Python interpreter and see if it matches your expectations:

```python
>>> chr(0)  
>>> print(chr(0))  
>>> "this is a test" + chr(0) + "string"  
>>> print("this is a test" + chr(0) + "string")
```

输出为空。


**2. Unicode Encodings**

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than  
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings? UTF-8 序列长度更短，压缩比更高，而且冗余字符少
(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string  
that yields incorrect results. 

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):  
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

- 英文在 UTF-8 中是一字节编码，范围在 0–127，是可以直接解码的。
- 例如中文在 UTF-8 中是三字节编码 0xE4 开头，必须要合法的 3 字节序列才能解码

(c) Give a two byte sequence that does not decode to any Unicode character(s). \xbd\xa0，选择中文三字节编码的后两个字节，就无法解码。

## Codes

BPE 的训练过程如下：

- 初始化词汇表 vocab
	- 将整型 0-255 映射到对应的 byte
	- 将 Special Token 加入 vocab，这样它就是不可分割的
- 将文本分为一个个 subword
	- 如果不存在 Special Token，直接用正则表达式分。
	- 如果存在 Special Token，就先根据 tok 分为一个个 part，然后在 part 中再分。
- 接下来就需要统计出现频率最高的 Pair Count，然后用这个 Pair 替换

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {}
    merges = []
    
    
    # 1. init vocab
    for i in range(256):
        bt = bytes([i])
        vocab[i] = bt
    for tok in special_tokens:
        # special_token 不可分割
        vocab[len(vocab)] = tok.encode("utf-8")
    
    # 2. read train data
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    subwords = []
    
    # 3. split text into subwords
    if special_tokens:
        toks = sorted(special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        parts = re.split(f"({union})", text)

        st = set(special_tokens)
        for part in parts:
            if not part or part in st:
                continue
            subwords.extend(re.findall(PAT, part))
    else:
        subwords = re.findall(PAT, text)
    
    # 4. count up pairs
    unicodes = [subword.encode("utf-8") for subword in subwords]
    
    while len(vocab) < vocab_size:
        # find pair with max count
        to_merge = (None, 0)
        pair_counts = defaultdict(int)
        for unicode in unicodes:
            for idx in range(len(unicode) - 1):
                pair = (bytes([unicode[idx]]), bytes([unicode[idx+1]]))
                pair_counts[pair] += 1
                
                if (to_merge[0] is None)  or \
                    (pair_counts[pair] > to_merge[1]) or \
                    (pair_counts[pair] == to_merge[1] and pair > to_merge[0]):
                    to_merge = (pair, pair_counts[pair])
        
        if to_merge[0] is None:
            raise
        
        # record
        new_unicodes = []
        new_token = to_merge[0][0] + to_merge[0][1]
        vocab[len(vocab)] = new_token
        merges.append(to_merge[0])
        
        # replace pair
        for idx, unicode in enumerate(unicodes):
            to_replace = [i for i in range(len(unicode)-1) if unicode[i:i+2] == to_merge[0]]
            if not to_replace:
                continue
            i = 0
            new_unicode = []
            while i < len(unicode):
                if i in to_replace:
                    new_unicode.append(new_token)
                    i += 2
                else:
                    new_unicode.append(unicode[i])
                    i += 1
            new_unicodes.append(new_unicode) 

        unicodes = new_unicodes
        
    return vocab, merges
```

> 在选择频率最高的 Pair 时候，如果出现两个频率一样的，就需要按照字典序选择一个最大的。但是这个字典序该怎么比呢？因为我们的 Pair 是 Byte Pair，也就是说比较的是 Byte 的字典序，那需要变成字符再比较吗？其实一个 Byte Pair 都不一定能转为 Character（Problem 2 里面提过），所以直接比较 Byte 字典序就好。

--- 

 朴素办法整体时间复杂度为 $O(V*T)$，它的开销主要在于：

1. 每次循环都要遍历一次文本，来得到最新的 Pair Counts
2. 得到需要 Merge 的 Pair 之后，又需要遍历一次文本，来替换 Pair

首先对于第一个问题，我们想要获得频率最高的 Pair 可以通过堆这个数据结构来实现，它获取堆顶元素并且更新堆的时间复杂度是 $O(logn)$。但是我们每次 Merge 之后需要更新 Pair 的上下文，那就需要对堆里面的每一个元素进行修改吗？其实可以通过懒更新的办法，假如需要 `pair_counts\[pair]++`，那我们可以把更新后的 Pair 直接 Push 到堆里面，然后在取堆顶的时候判断元素的合法性。

第二个问题，每次修改 Pair 也需要遍历一次文本有什么解决办法呢？如果采用 list 存储文本，每次 delete 的开销都是 $O(n)$，我们可以通过双向链表来存储，这样在已知节点位置的情况下，删除插入的时间复杂度就是 $O(1)$ 了。那我们该怎么知道节点位置呢？我们可以建立一个反向索引字典，记录每一个 token 在文本中哪些位置：token_id → set\[Node]。

> 之前我们区分 Subword 是通过二维数组进行，现在使用双向链表就需要在里面插入特殊节点来区分不同 Subword。

 从堆顶取出的合法的 Pair(a, b)，假设有 x → a → b → y，我们进行如下操作：

 1. pair_counts 把 (a, b) 这一项直接去掉
 2. 在反向链接中 key=a 和 key=b 的项 remove掉 符合 a -> b的 节点
 3. 如果 x 存在，把 (x, a)--，并且 (x, c) ++, 直接把(x, c)加入堆
 4. 如果 y 存在，那么 (b, y)-- (c, y)++  (c, y) 加入堆中（不需要管后面会不会更新，因为pop时候会检查，这样时间复杂度低）
 5. 将双向链表中 x -> a -> b -> y 变成 x -> c -> y
 6. 在反向链接中加入 key=c，记录位置  

```python
    heap: list[HeapItem] = []
    linkedlist = LinkedList[int]()
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    back_link: dict[int, set[LinkNode[int]]] = defaultdict(set)
    
    subwords = [list(subword.encode("utf-8")) for subword in subwords]
    for subword in subwords:
        for idx in range(len(subword) - 1):
            cnt, nxt = subword[idx], subword[idx+1]
            node = linkedlist.push_back(cnt)
            back_link[cnt].add(node)
            pair_counts[(cnt, nxt)] += 1
        linkedlist.push_back(subword[-1])
        linkedlist.push_back(-1)  # 插入一个分割符，来区分不同的 subword
    
    for (a, b), cnt in pair_counts.items():
        item = HeapItem(cnt, a, b, vocab)
        heapq.heappush(heap, item)
    
    while heap and len(vocab) < vocab_size:
        heapitem = heapq.heappop(heap)
        pair = heapitem.get_pair()
        
        # 采用懒删除，所以每次判断是不是存在的pair
        if pair not in pair_counts or pair_counts[pair] != heapitem.count:
            continue

        merges.append((vocab[pair[0]], vocab[pair[1]]))
        new_token = len(vocab)
        vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]
        
        # 修改双向链表，反向链接和pair_counts字典
        related_nodes = list(back_link[pair[0]])
        # --- 1 ---
        pair_counts.pop(pair)
        for node in related_nodes:
            if node.nxt is None or node.value != pair[0] or node.nxt.value != pair[1]:
                continue
            a, b = node, node.nxt
            x, y = a.pre, b.nxt
            
            # --- 2 ---
            back_link[pair[0]].discard(a)
            back_link[pair[1]].discard(b)
            # --- 3 ---
            if x is not None and x.value != -1:
                xa = (x.value, a.value)
                xc = (x.value, new_token)
                pair_counts[xa] -= 1
                pair_counts[xc] += 1
                heapq.heappush(heap, HeapItem(pair_counts[xa], *xa, vocab))
                heapq.heappush(heap, HeapItem(pair_counts[xc], *xc, vocab))
            # --- 4 ---
            if y is not None and y.value != -1:
                by = (b.value, y.value)
                cy = (new_token, y.value)
                pair_counts[by] -= 1
                pair_counts[cy] += 1
                heapq.heappush(heap, HeapItem(pair_counts[by], *by, vocab))
                heapq.heappush(heap, HeapItem(pair_counts[cy], *cy, vocab))
            # --- 5 ---
            a.value = new_token
            linkedlist.delete_node(b)
            # --- 6 ---
            back_link[new_token].add(a)
    
    return vocab, merges
```

这里需要注意 x → a 这个 Pair 也需要 Push 到堆中，因为 pair_counts\[(x,a)] 自减后下一轮也可能频率最高，如果不进行 Push，下一轮取出来（也许都取不出来，因为之前从堆里 Pop 了）合法性检查不通过，就得不到了。==懒加载的策略中，我们是通过 Push 对堆内元素进行更新的==。

```python
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PAS
================================= 3 passed in 80.31s =================================
```

>这里 related_nodes 是“过期快照”，for 循环中更新了 back_link 会不会导致取出不存在的节点呢？实际上不会，删除节点时候会把前驱后继设为 None，然后在取 node 时候会进行检查。


**在 Training Loop 章节有更优化的代码**

# BPE Tokenizer

Tokenizer 的功能就是接受训练得到的 vocab 和 merges，然后把输入文本转为对应的索引 id，或者把索引 id 转回文本，我们需要 implement 的函数包括 encode，decode 等等。但是第二部分作业坑非常多，后面会具体说。

根据之前 BPE Train 的优化思路，我们很容易能想到如何进行 Encode：

1. 用双向链表表示 Subword，用反向索引定位 bytes → Node 节点
2. 每次 Merge 通过反向索引找到符合的 Pair，然后在双向链表中把 a → b 变成 c，同时更新反向索引
3. 最后遍历双向链表就能得到 Merge 之后的文本了

于是就有了最初的代码：

```python
class BPE_Tokenizer:
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab.copy()
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.merges = merges
        self.rev_vocab = {v:k for k,v in vocab.items()}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.sp_union = "|".join(re.escape(t) for t in self.special_tokens)
      
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):

        # Load and normalize vocab: keys -> int, values -> bytes
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            raw_vocab = json.load(vf)

        norm_vocab = {v:k.encode("utf-8") for k, v in raw_vocab.items()}
        norm_merges = []
        # Load and normalize merges: ensure tuples of bytes
        for line in open(merges_filepath, "rb"):
            a, b = line.split()
            
            norm_merges.append(
                (
                    a.encode("utf-8") if isinstance(a, str) else a, 
                    b.encode("utf-8") if isinstance(b, str) else b
                )
            )

        return cls(norm_vocab, norm_merges, special_tokens)
     
     
    def encode(self, text: str) -> list[int]:
        subwords = []
        if self.special_tokens:
            subwords = re.split(f"({self.sp_union})", text)
        else:
            subwords = re.findall(self.PAT, text)
        
        ids = []
        for subword in subwords:
            ids.extend(self.merge_bytes([self.rev_vocab[bytes([byte_id])] for byte_id in subword.encode("utf-8")]))
        
        return ids
    
    def merge_bytes(self, raw_vocab_ids: list[int]) -> list[int]:
        merged_vocab_ids = []
        # 构建双向链表
        linkedlist = LinkedList[int]()
        # 构建一个反向链接
        back_link = defaultdict(list)
        for raw_vocab_id in raw_vocab_ids:
            node = linkedlist.push_back(raw_vocab_id)
            back_link[raw_vocab_id].append(node)
        
        # 遍历merges
        for merge in self.merges:
            new_token = merge[0] + merge[1]
            for node in list(back_link[self.rev_vocab[merge[0]]]):
                # 假设有 a → b
                a, b = node, node.nxt
                if b is None or self.vocab[b.value] != merge[1]:
                    continue
                
                # a → b 变成 c
                a.value = self.rev_vocab[new_token]
                linkedlist.delete_node(b)
                
                # 更新backlink
                back_link[self.rev_vocab[merge[0]]].remove(a)
                back_link[self.rev_vocab[merge[1]]].remove(b)
                back_link[self.rev_vocab[new_token]].append(a)
              
        head = linkedlist.head  
        while head is not None:
            merged_vocab_ids.append(head.value)
            head = head.nxt
        
        return merged_vocab_ids
             
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    
    def decode(self, ids: list[int]) -> str:
        return (b''.join([self.vocab[_id] for _id in ids])).decode("utf-8", errors="replace")

```

> handout 里面要求我们 implement `from_file` 这个方法，但实际上在 pytest 中没有进行测试。

运行 `uv run pytest tests/test_tokenizer.py` 直接报错了，提示没有 resource 这个库。我重新安装了一遍还是报这个错，查了百度才知道这个库在 linux 上面才能用，windows 不支持。所以我在 `test_tokenizer.py` 里面把需要用到 resource 库的地方都注释了，它主要用来监控内存占用，用处不大。

修改之后 pytest 还是报错了，我写了个 test 进行测试，发现 Tokenizer 里面的 vocab 居然没有 `b" "` 这一项，那肯定是 `from_file` 写错了。经过研究我才发现：==由于 json 文件不能存储 bytes，所以 vocab 和 merges 两个文件里面的 bytes 都是以 unicode 格式存储的==，所以我们读取 token 之后需要进行 unicode 和 bytes 的转换。

```python
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
    
      
class BPE_Tokenizer:
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        return cls(vocab, merges, special_tokens)
     
```

这次测试成功了，但是 pytest 还是全部 fail。经过排查最终发现，原来是 pytest 里面读取文件时候编码错误了，把 pytest 里面全部 open 加上 `encoding="utf-8"` 就.......还是fail了。

经过 debug 发现问题出在 Special Token 身上。在训练的时候，Special Token 不能参与合并，所以在处理的时候我们把它忽略了。==但是在 Encode 的时候，Special Token 也需要输出对应的索引号，也就是说我们不能忽略 Special Token了==。

```python
    def encode(self, text: str) -> list[int]:
        subwords = []
        if self.special_tokens:
            for subword in re.split(f"({self.sp_union})", text):
                subwords.extend([subword] if subword in self.special_tokens else re.findall(self.PAT, subword))
        else:
            subwords = re.findall(self.PAT, text)
        
        ids = []
        for subword in subwords:
            if subword in self.special_tokens:
                ids.append(self.rev_vocab[subword.encode("utf-8")])
            else:
                ids.extend(self.merge_bytes([self.rev_vocab[bytes([byte_id])] for byte_id in subword.encode("utf-8")]))
        
        return ids
```

这次就通过了，但是耗时还是比较高，是不是我们的想法太复杂了？

```python
=========================== 23 passed, 2 skipped in 67.75s (0:01:07) ============================
```

这个算法中每次都需要遍历 Merge 列表，但对于输入文本通常为一句话，Merges 长度在 10\^5 的情况下来说，开销太大了，我们应该优化遍历 Merge 的开销。这里我们参考 Train 阶段的思路，用小根堆和懒删除来优化遍历 Merges 的开销。

```python
    def encode(self, text: str) -> list[int]:
        subwords = []
        out = []
        if self.special_tokens:
            for subword in re.split(f"({self.sp_union})", text):
                subwords.extend([subword] if subword in self.special_tokens else re.findall(self.PAT, subword))
        else:
            subwords = re.findall(self.PAT, text)
        
        for subword in subwords:
            if subword in self.special_tokens:
                out.append(self.rev_vocab[subword.encode("utf-8")])
            else:
                out.extend(self.encode_non_special(subword))
        return out
    
    def encode_non_special(self, subword: str) -> list[int]:
        heap = []
        # 1. subword 变成 vocab-ids
        raw_vocab_ids = [self.rev_vocab[bytes([byte_id])] for byte_id in subword.encode("utf-8")]
        # 2. 用双向链表存文本
        linkedlist = LinkedList[LinkItem]()
        for raw_vocab_id in raw_vocab_ids:
            linkedlist.push_back(LinkItem(vocab_id=raw_vocab_id))
        
        # 小根堆，(rank, node)
        node = linkedlist.head
        while node is not None:
            rank = self.get_rank(node)
            if rank != INF:
                item = HeapItem(rank, node)
                heapq.heappush(heap, item)
            node = node.nxt
        
        
        # x → a → b
        while heap:
            heap_item = heapq.heappop(heap)
            a = heap_item.node
            b = a.nxt
                
            if not a.value.alive or b is None or heap_item.rank != self.get_rank(heap_item.node):
                continue
            
            pair = (self.vocab[a.value.vocab_id], self.vocab[b.value.vocab_id])
            new_token = pair[0] + pair[1]
            new_vocab_id = self.rev_vocab[new_token]
            
            # new_token 替代 (a, b)
            a.value = LinkItem(new_vocab_id)
            b.alive = False
            linkedlist.delete_node(b)
            
            
            x = a.pre
            if x is not None:
                if (rank := self.get_rank(x)) != INF:
                    heapq.heappush(heap, HeapItem(rank, x))
            
            if (rank := self.get_rank(a)) != INF:
                heapq.heappush(heap, HeapItem(rank, a))
            
        out = []
        node = linkedlist.head
        while node is not None:
            out.append(node.value.vocab_id)
            node = node.nxt
        return out
        
    
    def get_rank(self, node: LinkNode[LinkItem]) -> int:
        if not node.value.alive or node.nxt is None:
            return INF
        return self.merges_rank.get((self.vocab[node.value.vocab_id], self.vocab[node.nxt.value.vocab_id]), INF)
```

这个算法的思路就是，我们将字符串的每一个 Pair 都当做 Suspected Merge-Pair。假如这个 Pair 确实在 Merges 里面，就是说它可以被 Merge，那么就记录 (Rank, Pair\[0])，这就减少了遍历 Merges + Subword 寻找待 Merge Pair 的开销。因为采用懒删除，这时候我们从小根堆取出的 Pair 需要进行合法性评估，假如合法，那么它就是我们下一个需要 Merge 的 Pair 了。对于链表中 x → a → b，我们删除 b，将 a 的值改为 Merge 后的 new_token，之后更新 x 的 Pair 和 a 的 Pair。

```python
=================================== 23 passed, 2 skipped in 3.04s ===================================
```

# Torch Infra

## Linear

```python
class Linear(nn.Module):
    
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        """
        计算时候是 y=xW^T,存储的是 W 而不是 W^T,所以要反过来
        """
        super(Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand((out_features, in_features), **factory_kwargs))
        
        self.init_parameter()
    
    
    def init_parameter(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 这里是行向量，所以进行的变化是 y=xW^T
        """
        return torch.matmul(x, self.weight.T)
```

adapter 中需要注意：更新 module 参数时候不能直接 `linear.weight = weight`。

```python
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    from infra.linear import Linear
    
    device, dtype = in_features.device, in_features.dtype
    linear = Linear(d_in, d_out, device, dtype)
    linear.load_state_dict({'weight': weights})
    
    return linear.forward(x=in_features)
```

## Embedding

```python
class Embedding(nn.Module):
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: Optional[torch.device]=None, 
        dtype: Optional[torch.dtype]=None
    ) -> None:
        super(Embedding, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed = nn.Parameter(torch.rand((num_embeddings, embedding_dim), **factory_kwargs))
        self.init_parameter()
        
    def init_parameter(self):
        nn.init.trunc_normal_(self.embed, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: torch.Tensor):
        return self.embed[x]
```

```python
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    from infra.embedding import Embedding
    
    device, dtype = weights.device, weights.dtype
    embedding = Embedding(vocab_size, d_model, device, dtype)
    embedding.load_state_dict({"embed": weights})
    
    return embedding.forward(token_ids)
```

## RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}} * W
$$

```python
class RMSNorm(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        eps: float=1e-5,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(RMSNorm, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model), **factory_kwargs))
        
    
    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        norm = x / torch.sqrt(mean + self.eps) * self.weight
        return norm.to(in_type)
        
```

> `torch.mean` 需要设置 keepdim，否则形状为 \[batch_size, seq_len]，没办法广播和 x 相除。

```python
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from infra.rms_norm import RMSNorm
    
    device, dtype = weights.device, weights.dtype
    rms_norm = RMSNorm(d_model, eps, device, dtype)
    rms_norm.load_state_dict({"weight": weights})
    return rms_norm.forward(in_features)
```

## SwiGLU

$$
FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(SiLU(W_1) \odot W_3x)
$$

```python
class SwiGLU_FFN(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(SwiGLU_FFN, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
      
        self.d_ff = d_ff
        self.d_model = d_model
        
        self.w1 = nn.Parameter(torch.rand((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.rand((d_model, d_ff), **factory_kwargs))
        self.w3 = nn.Parameter(torch.rand((d_ff, d_model), **factory_kwargs))
        
        self.init_parameter()
        
    def init_parameter(self):
        nn.init.normal_(self.w1, mean=0, std=math.sqrt(2/self.d_ff))
        nn.init.normal_(self.w2, mean=0, std=math.sqrt(2/self.d_model))
        nn.init.normal_(self.w3, mean=0, std=math.sqrt(2/self.d_ff))
    
    
    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
    
    
    def forward(self, x: torch.Tensor):
        return (self.SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
        
```

```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from infra.swiglu import SwiGLU_FFN
    
    device, dtype = w1_weight.device, w1_weight.dtype
    swiglu = SwiGLU_FFN(d_model, d_ff, device, dtype)
    swiglu.load_state_dict({
        "w1": w1_weight,
        "w2": w2_weight,
        "w3": w3_weight
    })
    return swiglu.forward(in_features)
```

## RoPE

```python
class RoPE(nn.Module):
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device]=None
    ):
        super(RoPE, self).__init__()
        
        position = torch.arange(max_seq_len, device=device, dtype=torch.float32) # [max_len]
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        sinusoid = torch.outer(position, freq)
        rot = torch.exp(1j * sinusoid)  # [max_seq_len, dim//2]
        self.register_buffer("rot_cache", rot)
    
    def forward(self, qk: torch.Tensor, token_positions: torch.Tensor):
        *_, dim = qk.shape
    
        assert dim % 2 == 0, "dim must be even"
        
        qk_complex = qk.view(*qk.shape[:-1], dim//2, 2) # [bsize, nheads, seq_len, dim//2, 2]
        qk_complex = torch.view_as_complex(qk_complex)  # [bsize, nheads, seq_len, dim//2]
        
       
        rotated_qk_complex = qk_complex * self.rot_cache[token_positions]
        rotated_qk = torch.view_as_real(rotated_qk_complex)  # [bsize, nheads, seq_len, dim//2, 2]
        rotated_qk = rotated_qk.view_as(qk)
        
        return rotated_qk
```

## Softmax

$$
softmax(x_i)=\frac{e^{x_i}}{\sum_j^{dim}{e^{x_j}}}
$$
```python
def softmax(in_features: torch.Tensor, dim: int):
    max_features = in_features.max(dim=dim, keepdim=True).values
    exp = torch.exp(in_features - max_features)
    return exp / exp.sum(dim=dim, keepdim=True)
```

> softmax 中有指数运算，如果 $x_i$ 的值非常大可能导致指数爆炸，inf/inf 就会导致 nan 错误。为了解决这个问题，可以减去最大值，变成都是小等于 0 的数字。


## Scaled Dot Attention

```python
def scaled_dot_production_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor]=None
):
    *_, d_k = q.shape
    
    scores = q @ k.transpose(-1, -2) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask==False, float("-inf"))
    
    attn = softmax(scores, dim=-1)
    attn = attn @ v
    
    return attn
```

> 这里有几个小坑：
> 1. `scores.masked_fill_` 是原地操作，`scores.masked_fill` 需要赋值
> 2. 缩放点积注意力不需要处理注意力头的问题
> 3. 由于我们生成 mask 时候用的是 max_seq_len，所以进行掩码时候要进行截长。

```python
    *_, t_q, d_k = q.shape
    *_, t_k, d_k = k.shape   
    
    scores = q @ k.transpose(-1, -2) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask[:t_q, :t_k]==False, float("-inf"))
```

## MultiHead Attention

```python
class Multihead_Self_Attention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        device: Optional[torch.device] = None,
        **rope_config
    ):
        super(Multihead_Self_Attention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        if use_rope:
            self.rope = RoPE(**rope_config, device=device)

        self.q_weight = nn.Parameter(torch.rand((d_model, d_model), device=device))
        self.k_weight = nn.Parameter(torch.rand((d_model, d_model), device=device))
        self.v_weight = nn.Parameter(torch.rand((d_model, d_model), device=device))
        self.o_weight = nn.Parameter(torch.rand((d_model, d_model), device=device))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        token_positions: Optional[torch.Tensor] = None,
    ):

        q = x @ self.q_weight.T
        k = x @ self.k_weight.T
        v = x @ self.v_weight.T

        q = q.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attn = scaled_dot_production_attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view_as(x)
        return attn @ self.o_weight.T
```

为了加速计算，可以把 q,k,v,o 四个矩阵放在一起计算，用一个 `(d_model, 4*d_model)`。

# Transformer

## Transformer Block

```python
class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        use_rope: bool,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        eps: float = 1e-5,
        **rope_config
    ):
        super(TransformerBlock, self).__init__()
        
        self.attn = Multihead_Self_Attention(d_model, num_heads, use_rope, device, **rope_config)
        self.ffn = SwiGLU_FFN(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], token_positions: Optional[torch.Tensor]):
        attn_out = self.attn(self.ln1(x), mask, token_positions)
        residual = x + attn_out
        ffn_out = self.ffn(self.ln2(residual))
        return residual + ffn_out
```

写 `adapter.py` 时候把之前的代码改了一下，一开始我们实现了 `nn.Linear` 但是后面用的还是 `nn.Parameter`，并且我把变量名改成要求的格式。

```python
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from infra.transformer import TransformerBlock
    assert d_model % num_heads == 0

    device, dtype = in_features.device, in_features.dtype
    d_k = d_model // num_heads
    block = TransformerBlock(
        d_model,
        d_ff,
        num_heads, 
        use_rope=True, 
        device=device,
        dtype=dtype,
        theta=theta,
        max_seq_len=max_seq_len,
        d_k=d_k,
    )
    weights["attn.o_proj.weight"] = weights["attn.output_proj.weight"]
    block.load_state_dict(weights, strict=False)
    
    mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device))
    B, S, _ = in_features.shape
    token_positions = torch.arange(S, device=device).expand(B, -1)
    return block(in_features, mask, token_positions)
```

## Transformer LM

最后一步就是组装每一个组件，注意 Ttransformer 最后返回的是未 softmax 的分布：

```python
class Transformer(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        theta: float,
        d_k: int,
        eps: float = 1e-5,
        use_rope: bool = True,
    ):
        super(Transformer, self).__init__()

        self.device = device
        self.context_length = context_length
        
        
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
                TransformerBlock(d_model, d_ff, num_heads, use_rope, device, dtype, theta, d_k, context_length, eps)
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        B, L = x.shape
        positions = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        
        embed = self.token_embeddings(x)
        attn = embed
        for layer in self.layers:
            attn = layer(attn, positions)
        
        norm = self.ln_final(attn)
        proj = self.lm_head(norm)
        
        # return Softmax(proj, dim=-1)
        return proj
```

# Training

## Cross Entropy Loss

交叉熵损失 CrossEntropyLoss 公式为：

$$
l(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^{m}-\mathrm{log} p_{\theta}(x_{i+1}|x_{1:i})
$$

具体分为两个步骤：

1. 求 logits 的对数概率分布 log_softmax
2. 求负对数似然

---

**似然是什么？**

假设硬币正面朝上的概率是 $\theta$（模型参数），抛 10 次硬币，出现 “正正反正” 这样的结果（数据 D）的概率是：

$$
P(D|\theta) = \theta \times \theta \times (1-\theta) \times \theta = \theta^3(1-\theta)^1
$$

概率回答的是：已知参数 θ，结果 D 发生的可能性有多大？

现在反过来：我们已经抛了 10 次硬币，得到了 “正正反正” 这个结果（数据 D），想反推参数 θ（正面概率）应该取多少，才能让这个结果 “最合理”。似然函数就是把概率的自变量反过来：

$$
L(\theta|D) = P(D|\theta)
$$

似然回答的是：已知结果 D，参数 θ 取某个值时，能让这个结果发生的可能性有多大？我们的目标是找到让 ($L(\theta|D)$) 最大的 θ（最大似然估计）。

为了简化计算和方便反向传播，我们类似 softmax 的做法在似然函数前面加上 -log：

$$
-\log{L(θ∣D)}=-\log{P(D∣θ)}
$$

对于 Transformer 这样的语言模型，我们训练用 Teacher-Forcing，也就是说得到的结果就已经是 $P(\theta_{i+1}|\theta_{1:i})$ ，我们只需要把 logits 取对数概率，然后用 targets 当索引就能得到。与此同时考虑到 logits 第一个 dim 是 Batch，所以应该应该是对每一个 batch 的每一个 token 求负对数似然。

```python
def cross_entropy_with_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor,
    ignore_index: Optional[int] = None
):
    assert logits.dim() in (2, 3), "dim not match"
    assert targets.dim() in (1, 2), "dim not match"
    
    # log_prob: [batch_size, seq_len, vocab_size]
    log_prob = LogSoftmax(logits, dim=-1)
    _, vocab_size = log_prob.shape
    # log_prob_flat: [batch_size * seq_len, vocab_size]
    log_prob_flat = log_prob.reshape(-1, vocab_size)
    # targets_flat: [batch_size * seq_len]
    targets_flat = targets.reshape(-1)
    # tgt_log_prob: [batch_size * seq_len]
    tgt_log_prob = torch.gather(
        input=log_prob_flat,
        dim=1,
        index=targets_flat.unsqueeze(-1)
    ).squeeze(1)
    
    if ignore_index is not None:
        mask = (targets_flat != ignore_index)
        tgt_log_prob = tgt_log_prob[mask]
    
    return (-tgt_log_prob.sum() / mask.float().sum()) if ignore_index else -tgt_log_prob.mean()
    
```


## Optimizer

**动量**

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\end{align}
$$

**偏差修正**

$$
\begin{align}
\hat m_t &= \frac{m_t}{1-\beta_1^t} \\
\hat v_t &= \frac{v_t}{1-\beta_2^t} 
\end{align}
$$

**参数更新**

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

**AdamW 相对 Adam 在参数更新阶段进行衰减**

$$
\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\alpha \lambda \theta_{t-1}
$$

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            alpha = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                
                state["step"] += 1
                
                state["m"].mul_(beta1).add_(grad, alpha=1-beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                if group["correct_bias"]:
                    step_size = alpha * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                else:
                    step_size = alpha
                    
                denom = state["v"].sqrt().add_(eps)
                
                p.data.addcdiv_(state["m"], denom, value=-step_size)
                
                if wd != 0:
                    p.data.add_(p.data, alpha=-alpha * wd)
            

        return loss
```

## 学习率衰减

余弦退火算法进行学习率衰减：

$$
\alpha_t=
\begin{cases}
\frac{t}{T_w}\alpha_{max}, & t < T_W \\
\alpha_{min} + \frac{1}{2}(1 + \cos(\frac{t-T_w}{T_c-T_w}\pi))(\alpha_{max}-\alpha_{min}),  & x < 0 \\
\alpha_{min} , &t >T_c
\end{cases}
$$

```python
def lr_cosine_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_iters: int
):
    if t < warmup_iters:
        return t / warmup_iters * max_lr
    elif t <= cosine_iters:
        return min_lr + 0.5 * (1 + math.cos((t - warmup_iters) / (cosine_iters - warmup_iters) * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr
```


## 梯度裁剪

```python
def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    grads = []
    params_with_grad = []
    numels = []
    
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.view(-1))
            params_with_grad.append(param)
            numels.append(param.numel())
    
    if not grads:
        return
    
    grads = torch.cat(grads)
    l2_norm = torch.norm(grads, p=2)
    
    if l2_norm > max_l2_norm:
        grads *= max_l2_norm / (l2_norm + eps)
    
    pointer = 0
    for param, numel in zip(params_with_grad, numels):
        grad_chunk = grads[pointer: pointer + numel].view_as(param)
        param.grad.copy_(grad_chunk)
        pointer += numel
```

# Training Loop

这个部分就要求我们将所有组件拼在一起，写一个训练脚本训练 Transformer了，先看代码：

```python
from infra import (
    Transformer, 
    AdamW, 
    cross_entropy_with_logits, 
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
)
from bpe import train_bpe, BPE_Tokenizer
from pathlib import Path
import numpy as np
import argparse
import torch
import random
import pickle

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_memmap(path, dtype=np.uint16):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset


def train(args, train_data, val_data):
    assert args.dmodel % args.num_heads == 0
    
    device = torch.device(args.device)
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        dmodel=args.dmodel,
        d_ff=args.dff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=device,
        dtype=torch.float32,
        theta=args.theta,
        d_k=args.dmodel // args.num_heads,
        eps=args.eps,
        use_rope=args.use_rope
    ).to(device)
    optim = AdamW(model.parameters(), args.lr)
    
    best_losses = float("inf")
    for epoch in range(args.epochs):
        
        train_losses = []
        model.train()
        
        for iter_num in range(args.train_steps):
            
            lr = lr_cosine_schedule(
                iter_num,
                args.max_lr,
                args.min_lr,
                args.warm_up_it,
                args.cosine_it
            )
            
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            
            
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            
            optim.zero_grad()
            logits = model(x)
            loss = cross_entropy_with_logits(logits, y)
            loss.backward()
            gradient_clipping(model.parameters(), args.max_l2_norm)
            optim.step()
            train_losses.append(loss.item())

            print(f"epoch-{epoch} train loss: {train_losses[-1]:.4f}")
            
        if epoch % args.val_interval == 0:
            model.eval()
            
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    x, y = get_batch(val_data, args.batch_size, args.context_length, device)
                    logits = model(x)
                    loss = cross_entropy_with_logits(logits, y)
                    val_losses.append(loss.item())

            print(f"epoch-{epoch} validate loss: {val_losses[-1]:.4f}")


            if val_losses[-1] < best_losses:
                best_losses = val_losses[-1]
                filepath = Path(args.save_fp) / "checkpoint.pt"
                save_checkpoint(model, optim, iter_num, filepath)
                print("checkpoint saved")


def get_args():
    parser = argparse.ArgumentParser()
    # --- basic config ---
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_batches", type=int, default=100)
    # --- bpe ---
    parser.add_argument("--bpe_fp", type=str)
    parser.add_argument("--vocab_size", type=int, default=10000)
    # --- optim config ---
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warm_up_it", type=int, default=500)
    parser.add_argument("--cosine_it", type=int, default=10000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    # --- transformer config ---
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--dmodel", type=int, default=512)
    parser.add_argument("--dff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--theta", type=float, default=10000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--use_rope", type=bool, default=True)
    # --- filepath ---
    parser.add_argument("--train_dataset_fp", type=str)
    parser.add_argument("--validate_dataset_fp", type=str)
    parser.add_argument("--save_fp", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    import os

    print(os.getcwd())
    args = get_args()
    seed_everything(args.seed)
    
    VOCAB_FP = Path("vocab.pkl")
    MERGES_FP = Path("merges.pkl")
    special_tokens = ["<|endoftext|>"]  
    
    if VOCAB_FP.exists() and MERGES_FP.exists():
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open("merges.pkl", "rb") as f:
            merges = pickle.load(f)
    else:
        vocab, merges = train_bpe(args.bpe_fp, args.vocab_size, special_tokens)
        print("bpe train completed")
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open("merges.pkl", "wb") as f:
            pickle.dump(merges, f)
    
    tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
    
    TRAIN_ENCODE_FP = Path("train.dat")
    VAL_ENCODE_FP = Path("validate.dat")
    
    if not TRAIN_ENCODE_FP.exists() or not VAL_ENCODE_FP.exists():
        with open(args.train_dataset_fp, "r", encoding="utf-8") as f:
            train_text = f.read()
        train_encode_ids = np.array(tokenizer.encode(train_text)) 
        train_encode_ids.tofile(TRAIN_ENCODE_FP)
        
        with open(args.validate_dataset_fp, "r", encoding="utf-8") as f:
            val_text = f.read()
        val_encode_ids = np.array(tokenizer.encode(val_text))
        val_encode_ids.tofile(VAL_ENCODE_FP)
    
    
    train_data = get_dataset_memmap(TRAIN_ENCODE_FP)
    val_data = get_dataset_memmap(VAL_ENCODE_FP)

    print(f"Train data size: {len(train_data)} tokens")
    print(f"Val data size: {len(val_data)} tokens")
       
    train(args, train_data, val_data)
```

简单来说分为以下步骤：

1. 用训练集的 Corpus 训练 BPE
2. 用训练好的 BPE 对数据集进行 Encode，得到 train_ids 和 valid_ids，这样后面就不需要重复 Encode。
3. 进行 train loop

## 优化 train_bpe


然后第一个问题就出现问题了，虽然我的 train_bpe 通过了 pytest，但是在 TinyStoriesValid 数据集上进行训练时候预估要进行十几个小时，所以我需要对 train_bpe 进行修改\==。

我在 TinyStoriesValid 数据集上训练，将 vocab_size 设为 300 总耗时 5 分钟，通过 scalene 对性能进行分析得到以下信息：

![](http://img.xilyfe.top/img/bpe_scalene.png)

对时间进行降序，可以看到链表的操作耗时是最长的：==Python 中指针操作是非常耗时的，它与 C/C++ 不同，属性查找、指针解引用等等操作开销非常大==，所以我们考虑第一个优化方向是将链表变成简单的 list\[list\[int]]。

于是我删除了 back_link 和 linklist，用 `pair_to_indice: dict[tuple[int,int], tuple[int, int]]` 来存储。

```python
pair_counts: dict[tuple[int, int], int] = defaultdict(int)
pair_to_indice: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    

ids = [list(subword.encode("utf-8")) for subword in ids]
for i, token_ids in enumerate(ids):
    for j in range(len(token_ids)-1):
        pair = (token_ids[j], token_ids[j+1])
        pair_counts[pair] += 1
        pair_to_indice[pair].add((i, j))
```

merges 维护阶段的思路也参考之前的思路，对于 \[p, a, b, q] → \[p, X, q]，我们只需要更新 (p, a), (a, b), (b, q), (p, X), (X, q) 的 pair_counts 和 pair_to_indice。

```python
related_indices = pair_to_indice[pair].copy()
for i, j in related_indices:
    if ids[i][j] != pair[0] or len(ids[i]) <= j+1 or ids[i][j+1] != pair[1]:
        continue
            
    # === 1 ===
    if j > 0:
        old_prev_pair = (ids[i][j-1], ids[i][j])
        new_prev_pair = (ids[i][j-1], new_token_id)
        pair_counts[old_prev_pair] -= 1
        pair_counts[new_prev_pair] += 1
        pair_to_indice[old_prev_pair].discard((i, j-1))
        pair_to_indice[new_prev_pair].add((i, j-1))
        heapq.heappush(heap, HeapItem(pair_counts[old_prev_pair], *old_prev_pair, vocab))
        heapq.heappush(heap, HeapItem(pair_counts[new_prev_pair], *new_prev_pair, vocab))
    # === 2 ===
    if j + 2 > len(ids[i]):
        old_next_pair = (ids[i][j+1], ids[i][j+2])
        new_next_pair = (new_token_id, ids[i][j+2])
        pair_counts[old_next_pair] -= 1
        pair_counts[new_next_pair] += 1
        pair_to_indice[old_next_pair].discard(i, j+1)
        pair_to_indice[new_next_pair].add((i, j))
        heapq.heappush(heap, HeapItem(pair_counts[old_next_pair], *old_next_pair, vocab))
        heapq.heappush(heap, HeapItem(pair_counts[new_next_pair], *new_next_pair, vocab))
        
    # === 3 ===
    to_merge_token_ids = ids[i]
    new_token_ids = []
    i = 0
    while i < len(to_merge_token_ids):
        if i < len(to_merge_token_ids) - 1 and (to_merge_token_ids[i], to_merge_token_ids[i+1] == pair):
            new_token_ids.append(new_token_id)
            i += 2
        else:
            new_token_ids.append(to_merge_token_ids[i])
            i += 1
    ids[i] = to_merge_token_ids
del pair_counts[pair]
del pair_to_indice[pair]
```

但是这样的思路仍然存在一个问题：当我们修改 ids 之后，pair_to_indice 里面的索引全部偏移了（之前我们用的是节点所以没有这个问题）。最简单的解决思路是，那我们把存储的 (i, j) 变成存 i，然后去 ids\[i] 里面搜索行不行？这样子每次搜索就需要遍历 ids\[i]，不如直接把 ids\[i] 全部更新一遍。也就是说我们把 ids\[i] 里面每一个 pair 都去掉 counts 和 pair_to_indice，然后把 merge 之后新的 ids\[i] 里面每个 pair 再更新 counts 和 pair_to_indice。

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {}
    merges = []
    
    for i in range(256):
        bt = bytes([i])
        vocab[i] = bt
    for tok in special_tokens:
        # special_token 不可分割
        vocab[len(vocab)] = tok.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    ids = []
    if special_tokens:
        toks = sorted(special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        parts = re.split(f"({union})", text)

        st = set(special_tokens)
        for part in parts:
            if not part or part in st:
                continue
            ids.extend(re.findall(PAT, part))
    else:
        ids = re.findall(PAT, text)
        
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    # pair_to_indice 存储的是 pair 位于 subwords的[i] ,set[i]
    pair_to_indice: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    
    ids = [list(subword.encode("utf-8")) for subword in ids]
    for i, token_ids in enumerate(ids):
        for j in range(len(token_ids)-1):
            pair = (token_ids[j], token_ids[j+1])
            pair_counts[pair] += 1
            pair_to_indice[pair].add(i)
    
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        
        best_pair = max(pair_counts.items(), key=lambda x:(x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        related_indices = pair_to_indice[best_pair].copy()
        for i in related_indices:
            token_ids = ids[i]
            if len(token_ids) < 2:
                continue
            
            for old_pair in zip(token_ids[:-1], token_ids[1:]):
                pair_counts[old_pair] -= 1
                pair_to_indice[old_pair].discard(i)
            
            j = 0
            new_token_ids = []
            while j < len(token_ids):
                if j < len(token_ids) - 1 and (token_ids[j], token_ids[j+1]) == best_pair:
                    new_token_ids.append(new_token_id)
                    j += 2
                else:
                    new_token_ids.append(token_ids[j])
                    j += 1
            ids[i] = new_token_ids
            
            for new_pair in zip(new_token_ids[:-1], new_token_ids[1:]):
                pair_counts[new_pair] += 1
                pair_to_indice[new_pair].add(i)
             
    return vocab, merges
```

pytest 里面三个测试花费 17s 相较上个版本提高 76%，但是还能不能再优化呢？对 TinyStoriesValid 数据集再次跑了一轮 scalene，数据如下：

![](http://img.xilyfe.top/img/new_train_bpe_scalene.png)

开销主要在 pair_to_indice 字典还有初始化之后正则化的开销，pair_to_indice 的开销涉及重构算法思路了，但是 re 正则化还能优化：根据 pdf 里面指南，我们可以用多进程进行读取，然后用 re 分割文本。


| 读取文件耗时(s)             | TinyStoriesValid | TinyStoriesTrain |
| --------------------- | ---------------- | ---------------- |
| 单线程                   | 5.276s           |                  |
| 多线程(num_processes=8)  | 3.342s           |                  |
| 多线程(num_processes=16) | 2.899s           |                  |


## 优化 tokenizer encode


其次，在用 BPE 对数据集进行 Encode 的过程中出现了 Out of Memory 的错误，于是得找一个新的办法：==首先我们不应该从文件中直接读出全部文本，一行一行读并且 Encode 可以节省内存；其次再 buffer 中已经存了一定 encode_ids 就可以把他直接落盘==

```python
from bpe import BPE_Tokenizer
from pathlib import Path
import numpy as np 
import pickle


def encode_from_file(
    tokenizer: BPE_Tokenizer,
    in_filepath: str,
    out_filepath: str,
    chunk_size: int = 1024 * 1024
):
    in_path = Path(in_filepath)
    out_path = Path(out_filepath)
    assert in_path.exists()
    
    if not out_path.exists():
        with open(out_path, "wb"):
            pass
    
    buffer = []

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "ab") as fout:
        for i, line in enumerate(fin):
            ids = tokenizer.encode(line)
            buffer.extend(ids)
            
            if len(buffer) >= chunk_size:
                np.asarray(buffer, dtype=np.uint16).tofile(fout)
                buffer.clear()

            if i % 100_000 == 0:
                print("processed", i)
        
        
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(fout)
```

除此之外，Encode 里面也用了链表时间开销非常大：

![](http://img.xilyfe.top/img/scalene_tokenizer.png)

```python
def encode_non_special(self, subword: str) -> list[int]:
        tokens = [bytes([byte_id]) for byte_id in subword.encode("utf-8")]
        
        if len(tokens) == 1:
            return [self.rev_vocab[tokens[0]]]

        while len(tokens) >= 2:
            pairs = list(zip(tokens[:-1], tokens[1:]))
            pair = min(pairs, key=lambda p:self.merges_rank.get(p, INF))
            
            if pair not in self.merges_rank:
                break
            
            new_token = pair[0] + pair[1]
            
            i=0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return [self.rev_vocab[token] for token in tokens]
```

TinyStories_sample_5M 数据集上 Encode 耗时缩短了一半。

## 训练 Transformer

现在我们可以正式开始训练了。

# Generate Text

生成阶段没有太多说的，generate 时候将 token_ids 输入到模型，最后一个字符对应的 logits 就是预测的下一个 token。所以把它经过 softmax 得到概率分布，取出最大概率所在的索引，就能得到预测的 token。然后把预测的 token 加入 token_ids 不断重复的预测，就能得到生成的文本。最后当预测的字符为 EOS 或者生成文本长度达到上限就结束。

```python
def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: Transformer,
    tokenizer: BPE_Tokenizer,
    eos_token: Optional[str] = None
):
    token_ids = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        inp = torch.tensor(token_ids).unsqueeze(0)
        logits = model(inp).squeeze(0)
        probs = Softmax(logits, -1, temperature) # [seq_len, vocab_size]
        indices = probs.argmax(dim=-1) # [seq_len]
        next_token_id = indices[-1]
        
        if eos_token is not None and tokenizer.decode([next_token_id]).strip() == eos_token:
            break
        
        token_ids.append(next_token_id)
        
    return tokenizer.decode(token_ids)
```

用小型模型进行实验有时会生成质量很低的文本。两种简单的解码器技巧可以帮助解决这些问题。首先，在温度缩放中，我们用温度参数τ修改我们的softmax，其中新的softmax是解码器技巧 我们将用小型模型进行实验，而小型模型有时会生成质量很低的文本。两种简单的解码器技巧可以帮助解决这些问题。

1. 在温度缩放中，我们用温度参数 τ 修改 softmax。

$$softmax(v, \tau)_{i}=\frac {exp \left(v_{i} / \tau \right) }{\sum _{j=1}^{ | vocab \_size |} exp \left( v_{j} / \tau \right) }$$

2. 另一个技巧是 top-p 采样

$$
P\left(x_{t+1}=i | q\right)= \begin{cases}\frac{q_{i}}{\sum_{j \in V(p)} q_{j}} & if i \in V(p) \\ 0 & otherwise \end{cases}
$$

```python
def top_p_sampling(probs: torch.Tensor, p: float):
    sorted_probs, indices = torch.sort(probs, descending=True)
    accum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = accum_probs <= p
    
    # 保证至少有一个
    mask[..., 0] = True
    filitered_probs = sorted_probs * mask
    filitered_probs = filitered_probs / filitered_probs.sum()
    return indices[torch.multinomial(filitered_probs, num_samples=1).item()]


def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: Transformer,
    tokenizer: BPE_Tokenizer,
    eos_token: Optional[str] = None
):
    token_ids = tokenizer.encode(prompt)
  
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor(token_ids).unsqueeze(0)
            logits = model(inp)[0, -1, :]
            scaled_logits = logits / temperature
            probs = Softmax(scaled_logits, dim=-1)
            next_token_id = top_p_sampling(probs, top_p)
            
            if eos_token is not None and tokenizer.decode([next_token_id]).strip() == eos_token:
                break
            
            token_ids.append(next_token_id)
        
    return tokenizer.decode(token_ids)
```


最后 experiment 部分由于没有算力(~~懒得搞~~)，就不做了。