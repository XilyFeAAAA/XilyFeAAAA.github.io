---
title: CS224N Final Project-MinBert
date: 2025-12-11T11:24:11+08:00
authors:
  - Xilyfe
series:
  - CS224N
tags:
  - 深度学习
lastmod: 2026-01-30T02:47:27+08:00
featuredImage: http://img.xilyfe.top/img/20260130144509787.png
---

## Self-Attention

### BertModel.embed

```python
  def embed(self, input_ids):
    # input_ids: [bs, seq_len]
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = None
    ### TODO
    # [bs, seq_len, hidden_size]
    inputs_embeds = self.word_embedding(input_ids)

    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]

    pos_embeds = None
    ### TODO
    
    pos_embeds = self.pos_embedding(pos_ids)


    # Get token type ids, since we are not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    ### TODO
    
    embed = inputs_embeds + pos_embeds + tk_type_embeds
    embed = self.embed_layer_norm(embed)
    return self.embed_dropout(embed)
```

- `pos_ids` 是一个形如 \[\[1, 2, 3, ..., n], \[1, 2, 3, ..., n]] 的列表，形状和 `input_ids` 相同，用于获取位置嵌入，这里的 Position Embedding 用的是==可学习位置编码==。
- `tk_type_ids` 是用于区分句子的，但是这里没有用到，所以用的是 `torch.zeros`

### BertLayer

```python
  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    ### TODO
    output = dense_layer(output)
    output = dropout(output)
    context = input + output
    return ln_layer(context) 


  def forward(self, hidden_states, attention_mask):
    ### TODO

    # multi-head attention    
    attn_value = self.self_attention(hidden_states, attention_mask)
    
    # add-norm
    normed_attn = self.add_norm(hidden_states, attn_value, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

    # feedforward
    ff = self.interm_dense(normed_attn)
    ff = self.interm_af(ff)
    
    # add-norm
    normed_ff = self.add_norm(normed_attn, ff, self.out_dense, self.out_dropout, self.out_layer_norm)

    return normed_ff
```

BertLayer 就是==自注意力+残差归一+前馈网络+残差归一==的组合，没有特别需要注意的。


### BertSelfAttention

```python
  def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
    """
    softmax(QK^T/sqrt(d_k))V
    q, k, v 是 [bs, seq_len, hidden_state] 经过  transform() 变成 [bs, n_heads, seq_len, head_size]
    """
    bs, _, seq_len, _ = key.shape
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
      scores += attention_mask
    
    attn = torch.softmax(scores, dim=-1)
    attn = self.dropout(attn)    
    attn = torch.matmul(attn, value)
    # currently shape [bs, n_heads, seq_len, head_size]
    output = attn.transpose(1, 2).contiguous().view(bs, seq_len, self.all_head_size)
    
    return output
```

自注意力的实现也很简单，不过需要注意：这里 attention_mask 不是 0 和 1而是 -10000 和 0，所以直接加上去就好，不需要用 `scores.masked_fill`

## Optimizer

基础概念参见：[[optimizer]]

```python
	def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                ### TODO
                
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                
                state["step"] += 1
                state["m"] = group["betas"][0] * state["m"] + (1 - group["betas"][0]) * grad
                state["v"] = group["betas"][1] * state["v"] + (1 - group["betas"][1]) * grad ** 2

                if group["correct_bias"]:
                    m = state["m"] / (1 - group["betas"][0] ** state["step"])
                    v = state["v"] / (1 - group["betas"][1] ** state["step"])
                else:
                    m, v = state["m"], state["v"]
                
                
                p.data -= alpha * group["weight_decay"] * p.data 
                p.data -= alpha * m / (torch.sqrt(v) + group["eps"])
                
        return loss
```

PyTorch 的 Optimizer 基类中，超参数和 params 都存在 group 之中，m、v 保存在 state 中，一开始我的思路如上，但是这个代码存在以下问题：

1. Bias Correction 计算没有采用 Efficient Version，计算效率低
2. 我的计算每次都要创建新 Tensor，开销很大

- Efficient Bias Correction 的核心思想是：**简化公式优化掉 $\hat{m}$ 和 $\hat{n}$ 避免创建 Tensor 的开销**

$$
\begin{align}
\alpha\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} &= \frac{\alpha}{1-\beta_1^t}\cdot\frac{m}{\sqrt{v/(1-\beta_2^t)}} \\
&=\alpha \cdot\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}\cdot\frac{m}{\sqrt{v}}
\end{align}
$$

- 调用 PyTorch 自带的原地算子可以避免创建新 Tensor 的开销。

```python

    def step(self, closure: Callable = None):
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

                ### TODO
                
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                
                state["step"] += 1
                # state["m"] = group["betas"][0] * state["m"] + (1 - group["betas"][0]) * grad
                # state["v"] = group["betas"][1] * state["v"] + (1 - group["betas"][1]) * grad ** 2
                
                state["m"].mul_(beta1).add_(grad, alpha=1-beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                if group["correct_bias"]:
                    # m = state["m"] / (1 - group["betas"][0] ** state["step"])
                    # v = state["v"] / (1 - group["betas"][1] ** state["step"])
                    step_size = alpha * .sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                else:
                    # m, v = state["m"], state["v"]
                    step_size = alpha
                    
                denom = state["v"].sqrt().add_(eps)
                
                p.data.addcdiv_(state["m"], denom, value=-step_size)
                
                if wd != 0:
                    p.data.add_(p.data, alpha=-alpha * wd)
            

        return loss

```


## Classifier

`classifier.py` 这个任务主要是一个情感分类器，它基于之前写好的预训练的 Bert。任务分为两个：

1. 冻结 Bert 参数对分类头进行微调
2. 解冻参数，对全模型进微调

### BertSentimentClassifier

```python
class BertSentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        ### TODO
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)
        self.dense = torch.nn.Linear(in_features=config.hidden_size, out_features=128)
        self.classifier = torch.nn.Linear(in_features=128, out_features=config.num_labels)
        self.relu = torch.nn.ReLU()


    def forward(self, input_ids, attention_mask):        
        bert_output = self.bert(input_ids, attention_mask)
        # shape [bs, hidden_size]
        hidden_state = bert_output["pooler_output"]
        output = self.dense(hidden_state)
        output = self.dropout(output)
        output = self.relu(output)
        return self.classifier(output)
```

- 第一次设计时候只有一个线性层，我把 hidden_size 直接投影到 num_labels，但是准确率没有达到 handout 的要求。
- 第二次设计我用两个线性层，首先把 hidden_size 投影到 128 维，然后再投影到 num_labels，结果出现了训练集准确率变高，测试集正确率变低，明显是过拟合了。
- 第三次我先把 hidden_size 投影到 64 维然后再投影到 num_labels，过拟合问题解决掉了，但是 10 轮下来正确率还是没达到要求
- 第四次我仔细检查了代码，发现默认情况下训练的 lr 是 1e-5 也太低了，我改成 1e-3 就过了。


## MultiTaskClassifier

baseline 版本我没有对代码进行过多修改，我补全了 `MultitaskBERT` 的代码，在冻结了的 minbert 基础上微调三个分类头。考虑到任务之间的关系，嵌入过程中可能存在一些共享权重或结构，因此将 minBERT 模型与任务头一起进行 full-model 微调可能会全面提升性能。在实验中，每个 epoch 期间处理这些任务的顺序有所不同，从而产生了6种不同的 Sequential Learning。最后考虑到按顺序对任务进行训练可能导致后训练任务的梯度影响已训练的任务，所以采用了混合训练。

### 基准

#### 情感预测

情感预测的 forward 和 train 部分代码和 `classifier.py` 一致没有进行优化，但是出现了这个问题：**训练过程中从 epoch 2 开始 loss 不变了**。检查代码之后发现问题出在：

```python
def predict_sentiment(self, input_ids, attention_mask):
    bert_hidden_state = self.forward(input_ids, attention_mask)
    senti = self.senti_dense(bert_hidden_state)
    senti = self.dropout(senti)
    senti = self.senti_classifier(senti)
    return self.senti_relu(senti)
```

原因在于我把激活函数 ReLU 放在前向计算的最后一步了，ReLU 的函数表达式为 $ReLU=max(0,x)$，所以放在最后一步会固定把负数变成 0，正确的顺序应该是 ==Linear → ReLU → Dropout → ... → 最后一层 Linear→ 输出==：

```python
def predict_sentiment(self, input_ids, attention_mask):
    bert_hidden_state = self.forward(input_ids, attention_mask)
    senti = self.senti_dense(bert_hidden_state)
    senti = self.senti_relu(senti)
    senti = self.dropout(senti)
    return self.senti_classifier(senti)
```

#### 转述预测

这个任务是给定两个句子，判断是否含义相同，也就是说我会得到两个 input，那么该如何将它组装到一起呢？我的第一个想法是直接拼凑为 \[batch_size, seq_len, hidden_size * 2]，然后后续在把他投影到低维空间。

```python
def predict_paraphrase(self,
                      input_ids_1, attention_mask_1,
                      input_ids_2, attention_mask_2):
        para_1 = self.forward(input_ids_1, attention_mask_1)
        para_2 = self.forward(input_ids_2, attention_mask_2)        
        para_combined = torch.cat([para_1, para_2], dim=1)
        para_combined = self.para_dense(para_combined)
        para_combined = self.dropout(para_combined)
        para_combined = self.relu(para_combined)
        return self.para_classifier(para_combined)
```

二分类问题，forward 输出的 logits 形状为 \[B, 1], 还需要通过 sigmoid 转换为概率。计算 loss 时候 BCEWithLogitsLoss 自带 sigmoid 了，但是计算 acc 时候是通过 `torch.mean(pred\==labels)`，所以需要提前加 sigmoid。

#### 相似度预测

第三个任务是对两个句子的相似度进行打分，打分区间在 0-5。我的做法是将隐藏状态投影到一维空间 ，然后对 logits 它进行 sigmoid 就能得到 0-1 区间的概率，再乘上 5 就能得到 0-5 的相似分数了。

```python
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        simi_1 = self.forward(input_ids_1, attention_mask_1)
        simi_2 = self.forward(input_ids_2, attention_mask_2)
        simi_combined = torch.cat([simi_1, simi_2], dim=1)
        simi_combined = self.sts_dense(simi_combined)
        simi_combined = self.dropout(simi_combined)
        simi_combined = self.relu(simi_combined)
        return self.sts_classifier(simi_combined)
```

需要注意的是，根据注释 - 这个函数需要返回的是 **logits**，而不需要对它进行处理。观察 `evalution.py` 文件也可以看到，这里计算 `np.corrcoef` 是直接把 logits 放进去的。

```python
for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
	logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
	y_hat = logits.flatten().cpu().numpy()
	b_labels = b_labels.flatten().cpu().numpy()
	sts_y_pred.extend(y_hat)
	sts_y_true.extend(b_labels)
	sts_sent_ids.extend(b_sent_ids)
pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
```

> 皮尔逊相关系数反映的是**两个变量的线性相关关系**，跟值的绝对范围没关系，只要两个变量是一组一组对应的，并且是连续型的，都可以直接算，也就是说==不需要变化到 0-5 区间==，sigmoid 会降低原始 logit 与标签之间的线性相关性。

然而我们用均方差计算 loss 时候需要用 sigmoid 把它变化到 0-5 区间，使其落在标签分数同一数值域。

#### 实验结果


| bert + final-layer-finetune   | 验证集   | 测试集   |
| ----------------------------- | ----- | ----- |
| Stanford Sentiment Treebank   | 0.400 | 0.395 |
| Quora Dataset                 | 0.650 | 0.644 |
| SemEval STS Benchmark Dataset | 0.237 | 0.212 |

#### 改进

- [ ] 在训练期间不仅仅对三个数据集分别进行训练，而是进行全模型微调
- [ ] 如果全模型微调，不平衡数据如何处理？
- [x] 对比平均池化和 CLS 池化，看看哪个效果更好。
- [x] 怎么处理 pair 拼接，单纯 concat 到最后一个维度，还是有更好的办法？
- [ ] cross-encoder 加上 para_type_id
- [ ] STS 通过余弦相似度计算，而不是线性变换投影
- [ ] 用 3 个任务的数据集进行 MLM 预训练
- [ ] autocast(enabled=args.use_amp)和scaler.scale
- [ ] Gradient Surgery - PCGrad

### 平均池化

```python
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        token_embeddings = output["last_hidden_state"]
        expand_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * expand_mask, dim=1) / torch.clamp(expand_mask.sum(1), 1e-9)
```

需要注意：在进行平均池化时候需要考虑到输入文本长度不均，所以要用掩码去除无用的 embedding。我的办法是把 0/1 的掩码矩阵扩张到 \[batch_size, seq_len, hidden_size] 然后相乘，这样 0 位置的 token 就不会参与计算，实验结果如下：

| 数据集                     | SST 测试集 | Quora 测试集 | STS 测试集 |
| ----------------------- | ------- | --------- | ------- |
| baseline                | 0.395   | 0.644     | 0.212   |
| baseline + mean-pooling | 0.437   |           | 0.424   |


### Pair Sentence 拼接

#### DifProd 

参考 Sentence-BERT 的做法，将平均池化后得到的 u,v 拼接为 (u, v, |u - v|, u\*v)，然后接线性层分类。

```python
def predict_similarity(self,input_ids_1, attention_mask_1,input_ids_2, attention_mask_2):
    simi_1 = self.forward(input_ids_1, attention_mask_1)
    simi_2 = self.forward(input_ids_2, attention_mask_2)
    simi_combined = torch.cat([simi_1, simi_2, torch.abs(simi_1 - simi_2), simi_1 * simi_2], dim=1)
    simi_combined = self.sts_dense(simi_combined)
    simi_combined = self.dropout(simi_combined)
    simi_combined = self.relu(simi_combined)
    return self.sts_classifier(simi_combined)
```


#### Cross-Encoder

Cross-Encoder 的思路是将句子 A 和 B 的 embedding 拼接为 \[CLS] A \[SEP] B \[SEP]，然后把这个长序列送入模型。

```python
    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        input_ids = torch.cat([input_ids_1, input_ids_2[:, 1:]], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2[:, 1:]], dim=1)
        token_emb = self.forward(input_ids, attention_mask)
        simi_combined = self.sts_dense(token_emb)
        simi_combined = self.dropout(simi_combined)
        simi_combined = self.relu(simi_combined)
        return self.sts_classifier(simi_combined)
```

由于 Quora 数据集太大，这次仅用 STS 数据集进行测试，结果如下：

| 模型                                              | STS 验证集 | STS 测试集 |
| ----------------------------------------------- | ------- | ------- |
| baseline                                        | 0.237   | 0.212   |
| mean-pooling                                    | 0.437   | 0.424   |
| mean-pooling + difprod(u, v, \|u - v\|, u \* v) | 0.744   | 0.728   |
| mean-pooling + difprod(\|u - v\|, u \* v)       | 0.746   | 0.724   |
| mean-pooling + cross-encoder                    | 0.774   | 0.758   |

### 全模型微调

全模型微调和仅仅微调分类头不同，需要解冻 bert 模型的参数，也就是说：假如先对 SST 数据集进行训练，再对 Quora 数据集进行训练，那么后训练的数据集可能会对 SST 训练结果产生影响。我有三种想法进行训练：

1. Sequential Fintune：按照不同的顺序在每个 epoch 对三个任务轮流进行训练
2. Min-Anneal Sampling Finetune：根据 min-anneal 概率采样任务，每个轮次仅训练一个任务
3. Mixed Finetune：每个 epoch 对三个任务一起训练

#### Sequential Finetune

考虑到 Quora 数据集的大小比其他两个数据集大了两个数据集，会对模型产生较大影响，所以先对它进行训练，顺序为 Quora → SST → STS。

```python
		labels = ["sst", "para", "sts"]
        best_score = 0
        model.train()
        for epoch in range(args.epochs):
            train_loss = {label: 0 for label in labels}
            num_batches = {label: 0 for label in labels}
            
            # para
            for batch in tqdm(para_train_dataloader, desc=f'para-train-{epoch}', disable=TQDM_DISABLE):
                
                b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                b_labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.view(-1).float(), reduction='mean')

                loss.backward()
                optimizer.step()

                train_loss["para"] += loss.item()
                num_batches["para"] += 1
            
            train_loss["para"] /= num_batches["para"]
                
            # sst
            for batch in tqdm(sst_train_dataloader, desc=f'sst-train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
                loss.backward()
                optimizer.step()

                train_loss["sst"] += loss.item()
                num_batches['sst'] += 1

            train_loss["sst"] /= num_batches["sst"]

            # sts
            for batch in tqdm(sts_train_dataloader, desc=f'sts-train-{epoch}', disable=TQDM_DISABLE):
                
                b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                b_labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                prob = F.sigmoid(logits) * 5
                loss = F.mse_loss(prob.view(-1), b_labels.float(), reduction="sum") / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss["sts"] += loss.item()
                num_batches['sts'] += 1

            train_loss["sts"] /= num_batches["sts"]
            
            (para_train_acc, _, _, sst_train_acc,_, _, sts_train_corr, _, _) = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
            (para_dev_acc, _, _, sst_dev_acc,_, _, sts_dev_corr, _, _) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

            dev_score = (sst_dev_acc + para_dev_acc + sts_dev_corr) / 3.0
            if dev_score > best_score:
                best_score = dev_score
                save_model(model, optimizer, args, config, args.filepath)
            
            print(f"Epoch {epoch}: para losses {train_loss['para']}, sst losses {train_loss['sst']}, sts losses {train_loss['sts']}")
            print(f"Epoch {epoch}: sst train acc {sst_train_acc:.4f}, para train acc {para_train_acc:.4f}, sts train corr {sts_train_corr:.4f}")
            print(f"Epoch {epoch}: sst dev acc {sst_dev_acc:.4f}, para dev acc {para_dev_acc:.4f}, sts dev corr {sts_dev_corr:.4f}")

```


#### Min-Anneal Sampling Finetune

退火采样的思路在于：

- 如果按数据集大小比例采样，小任务永远得不到充分训练，会被大任务淹没。
- 如果简单地均匀采样任务（1/3 机会采样 A, B, C），小任务会被过度训练，而大任务的表示学习不足。

因此需要一个折中方法：**训练早期要更平衡（均匀）地选任务，后期逐渐让采样概率退火到按数据集大小分布。**

假设你有 n 个任务，数据集大小分别为：

$$
|D_1|, |D_2|, ..., |D_n|
$$

先计算**按大小的采样分布**：

$$
p_i^{size} = \frac{|D_i|}{\sum_j |D_j|}
$$

再计算**均匀分布**：

$$
p_i^{uniform} = \frac{1}{n}
$$

Min-Anneal Sampling 构建一个**随训练进展退火**的采样概率：

$$
p_i(t) = \min\left(1,\; \frac{t}{T_{anneal}}\right) \cdot p_i^{size} \;+\; \left(1 -  \min\left(1,\frac{t}{T_{anneal}}\right)\right) \cdot p_i^{uniform}
$$

代码如下：

```python
def min_anneal_prob(dataset_sizes, epoch, T_anneal):
    size_prob = dataset_sizes / dataset_sizes.sum()
    uniform_prob = np.ones_like(dataset_sizes) / len(dataset_sizes)
    alpha = min(1.0, epoch / T_anneal)
    return alpha * size_prob + (1 - alpha) * uniform_prob
    
if args.option == "finetune" and args.method == "sampling":
        best_score = 0
        task_labels = ["sst", "para", "sts"]
        dataset_size = torch.tensor([len(sst_train_dataloader), len(para_train_dataloader),  len(sts_train_dataloader)])   

        for epoch in range(args.epochs):
            
            model.train()            
            train_loss = 0
            train_batch = 0
            T_anneal = 4
            
            sst_iter = iter(sst_train_dataloader)
            para_iter = iter(para_train_dataloader)
            sts_iter = iter(sts_train_dataloader)
            num_iters = int(max(dataset_size))
            task_probs = min_anneal_prob(dataset_size, epoch, T_anneal)
            
            
            for _ in tqdm(range(num_iters), desc=f'sampling-train-{epoch}', disable=TQDM_DISABLE):
                loss = None
                task = np.random.choice(task_labels, p=task_probs)
                optimizer.zero_grad()
                if task == "sst":
                    try:
                        batch = next(sst_iter)
                    except StopIteration:
                        sst_iter = iter(sst_train_dataloader)
                        batch = next(sst_iter)
                    b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            
                elif task == "para":
                    try:
                        batch = next(para_iter)
                    except StopIteration:
                        para_iter = iter(para_train_dataloader)
                        batch = next(para_iter)
                    b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                    b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                    b_labels = batch["labels"].to(device)
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.view(-1).float(), reduction='mean')
                    
                elif task == "sts":
                    try:
                        batch = next(sts_iter)
                    except StopIteration:
                        sts_iter = iter(sts_train_dataloader)
                        batch = next(sts_iter)
                    b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                    b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                    b_labels = batch["labels"].to(device)
                    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    prob = torch.sigmoid(logits) * 5
                    loss = F.mse_loss(prob.view(-1), b_labels.float(), reduction="mean")
                    
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batch += 1
            
            train_loss /= train_batch
            
            if epoch % 2 == 0:
                para_dev_acc, _, _, sst_dev_acc,_, _, sts_dev_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

                dev_score = (sst_dev_acc + para_dev_acc + sts_dev_corr) / 3.0
                if dev_score > best_score:
                    best_score = dev_score
                    save_model(model, optimizer, args, config, args.filepath)
                
                print(f"Epoch {epoch}: loss {train_loss:.4f} sst dev acc {sst_dev_acc:.4f}, para dev acc {para_dev_acc:.4f}, sts dev corr {sts_dev_corr:.4f}")
```

#### Mixed Finetune

在一个 iteration 中对三个任务一起训练，得到三个损失后混合在一起：

$$
\hat{l} = \sum_{i}{\lambda_i*loss_i}
$$

loss 的权重可以设置为超参数或者让模型自学习，不过这里为了方便直接令其为 1。

但是将三个任务的 loss 混合在一起计算看似解决了 Sequential Finetune **后训练任务影响前训练任务**的问题，实际上还是一样，因为根本问题出在**一个任务的更新会拉参数远离另一个任务的最优解，导致收敛慢、性能不均**，三个 loss 累加求导并不能解决。解决方法是**梯度手术**：在反向传播后先“检查”它们的梯度是否冲突，如果冲突，就投影掉冲突部分，只保留有助于该任务的“有益”方向。

**PCGrad**

PCGrad 的思路很简单：对于不同的任务，我们将他们各自 backward 之后得到的梯度 flatten 成一个向量 grad_flatten。若不同任务的 grad_flatten 正交，那么就把任务 i 的梯度向量投影到与任务 j 的梯度向量正交的投影空间。

$$
\tilde{g}_i \leftarrow g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j
$$

```python
class PCGrad:
    
    def __init__(
        self,
        optimizer: Optimizer,
        eps: float = 1e-9
    ):
        self.optim = optimizer
        self.eps = eps
        self.params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    
    def zero_grad(self) -> None:
        self.optim.zero_grad()
    
    
    def step(self) -> None:
        self.optim.step()
    
    
    def gather_grad(self) -> torch.Tensor:
        grad = []
        for param in self.params:
            if param.grad is None:
                grad.append(torch.zeros_like(param).view(-1))
            else:
                grad.append(param.grad.view(-1).clone())
        return torch.cat(grad)
    
    def set_grad(self, grads) -> None:
        pointer = 0
        for param in self.params:
            length = param.numel()
            grad = grads[pointer : pointer + length].view_as(p)
            if p.grad is None:
                p.grad = grad.clone()
            else:
                p.grad.copy_(grad)
            pointer += length
    
    def backward(self, loss_lst):
        """
        1. 对每个loss进行回传
        2. 把每个loss的梯度flatten记录起来
        3. 两重for循环得到任务i 对其他任务j 梯度的投影
        4. 把投影后的梯度设置回去
        """
        grads = []
        proj_grads = []
        for loss in loss_lst:
            self.zero_grad()
            loss.backward()
            grad = self.gather_grad()
            grads.append(grad)
        
        num_loss = len(losses)
        
        for i in range(num_loss):
            grad_i = grads[i]
            js = range(num_loss)
            js.pop(i)
            for j in js:
                grad_j = grad[j]
                ij_dot = torch.dot(grad_i, grad_j)
                if dot < 0:
                    j_dot = torch.dot(grad_j, grad_j)
                    if j_dot > 0:
                        grad_i = grad_i - (ij_dot / (j_dot + self.eps)) * grad_j
            proj_grads.append(grad_i)
        
        proj_grads = torch.stack(proj_grads).mean()
        self.zero_grad()
        self.set_grad(proj_grads)
```

```python
best_score = 0
        dataset_size = torch.tensor([len(sst_train_dataloader), len(para_train_dataloader),  len(sts_train_dataloader)])   

        for epoch in range(args.epochs):
            
            model.train()
            
            sst_iter = iter(sst_train_dataloader)
            para_iter = iter(para_train_dataloader)
            sts_iter = iter(sts_train_dataloader)
            num_iters = int(max(dataset_size))
            
            
            for _ in tqdm(range(num_iters), desc=f'sampling-train-{epoch}', disable=TQDM_DISABLE):
                pc.zero_grad()
                
                # sst
                try:
                    batch = next(sst_iter)
                except StopIteration:
                    sst_iter = iter(sst_train_dataloader)
                    batch = next(sst_iter)
                b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                logits = model.predict_sentiment(b_ids, b_mask)
                sst_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            
                # para
                try:
                    batch = next(para_iter)
                except StopIteration:
                    para_iter = iter(para_train_dataloader)
                    batch = next(para_iter)
                b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                b_labels = batch["labels"].to(device)
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                para_loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.view(-1).float(), reduction='mean')
                
                # sts
                try:
                    batch = next(sts_iter)
                except StopIteration:
                    sts_iter = iter(sts_train_dataloader)
                    batch = next(sts_iter)
                b_ids_1, b_mask_1 = batch["token_ids_1"].to(device),  batch['attention_mask_1'].to(device)
                b_ids_2, b_mask_2 = batch["token_ids_2"].to(device),  batch['attention_mask_2'].to(device)
                b_labels = batch["labels"].to(device)
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                prob = torch.sigmoid(logits) * 5
                sts_loss = F.mse_loss(prob.view(-1), b_labels.float(), reduction="mean")
            
                pc.backward([sst_loss, para_loss, sts_loss])
                pc.step()

            
            para_dev_acc, _, _, sst_dev_acc,_, _, sts_dev_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

            dev_score = (sst_dev_acc + para_dev_acc + sts_dev_corr) / 3.0
            if dev_score > best_score:
                best_score = dev_score
                save_model(model, optimizer, args, config, args.filepath)
            
            print(f"Epoch {epoch}: loss {train_loss:.4f} sst dev acc {sst_dev_acc:.4f}, para dev acc {para_dev_acc:.4f}, sts dev corr {sts_dev_corr:.4f}")
```

测试结果如下：

| 模型                                      | SST 测试集 | Quora 测试集 | STS 测试集 |
| --------------------------------------- | ------- | --------- | ------- |
| baseline(Final Layer Finetune)          | 0.395   | 0.644     | 0.212   |
| full-model Sequential Finetune          | 0.253   | 0.6247    | 0.025   |
| full-model Min-Anneal Sampling Finetune |         |           |         |
| full-model Mixed Finetune               |         |           |         |
