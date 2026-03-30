---
title: CLIProxyAPI&大模型反代
date: 2026-03-30T19:22:18+08:00
featuredImage: http://img.xilyfe.top/img/20260330192432528.png
authors:
  - Xilyfe
series: []
tags:
  - 大模型
lastmod: 2026-03-30T08:54:15+08:00
---
{{< admonition type=info title="引言">}} 
进入 vibe coding 时代之后，大模型 token 的消耗速度成指数上涨。对于普通开发者来说，不管是在官网购买 token 还是在国内中转站购买 token 价格都很难承担。为了能白嫖各种模型，我试过 chat2api、grok2api 等各种反代项目，不管风控强容易失效而且部署很麻烦。今天发现 CLIProxyAPI 之后确实是非常之好用，记录一下部署中遇到的若干问题。
{{< /admonition >}}

## 1. CLIProxyAPI

### 1.1 前言 
>CLIProxyAPI 是一个为CLI提供 OpenAI/Gemini/Claude/Codex 兼容 API 接口的代理服务器。使我们可以使用本地或多账户的CLI方式，通过任何与 OpenAI / Gemini /Claude 兼容的客户端和 SDK 进行访问。

CLIProxyAPI 除了可以聚合这些 API 接口，担任一个 API 接口的管理者，它还可以对 OpenAI 等大模型进行反代。大模型的聊天界面（例如 GPT/Claude 的网页版）通常都有免费的额度，它的本质上也是在调用后端 API。所以可以通过抓包分析，找出界面发送请求的格式，通过模拟这些请求我们就可以实现用 api 进行请求一样的效果。

### 1.2 安装配置

windows 版本可以直接在 github 仓库里面下载：

![image.png](http://img.xilyfe.top/img/20260330195054007.png)

应该是命名错误了，12.5MB 的版本是 x64 也就是 64 位版本，11.4MB 的是 x86版本，所以我们需要下载前者。下载后解压安装包，我们会得到如下两个文件：

```
- config.example.yaml
- cli-proxy-api.exe
```

第一个是项目的配置文件，第二个就是可执行文件。我们需要把配置文件复制一份，重命名为 `config.yaml` 然后进行如下修改：

```yaml
# 服务器绑定主机/接口，默认空字符串同时绑定 IPv4/IPv6。
# 使用 "127.0.0.1" 或 "localhost" 可限制仅本机访问。
host: ""

# 服务器端口
port: 8317

# TLS 设置：启用后使用提供的证书与私钥监听 HTTPS。
tls:
  enable: false
  cert: ""
  key: ""

# 管理 API 设置
remote-management:
  # 是否允许远程（非 localhost）访问管理接口。
  # 为 false 时仅允许 localhost，仍需管理密钥。
  allow-remote: false

  # 管理密钥。若填写明文，启动时会自动哈希后生效。
  # 所有管理请求（包括本地）都需要该密钥。
  # 留空则完全禁用管理 API（所有 /v0/management 路由返回 404）。
  secret-key: ""

  # 为 true 时禁用内置管理面板资源下载与路由。
  disable-control-panel: false

  # 管理面板的 GitHub 仓库，可填写仓库 URL 或 releases API URL。
  panel-github-repository: "https://github.com/router-for-me/Cli-Proxy-API-Management-Center"

# 认证目录（支持 ~ 展开为主目录）
auth-dir: "~/.cli-proxy-api"

# 用于请求认证的 API 密钥
api-keys:
  - "your-api-key-1"
  - "your-api-key-2"
  - "your-api-key-3"

# 是否启用调试日志
debug: false

# 为 true 时禁用高开销 HTTP 中间件以降低高并发下的内存占用
commercial-mode: false

# 为 true 时将应用日志写入滚动文件而非 stdout
logging-to-file: false

# 日志目录的最大总大小（MB）；超过后会删除最旧的日志。0 表示不限制。
logs-max-total-size-mb: 0

# 为 false 时禁用内存用量统计聚合
usage-statistics-enabled: false

# 代理地址。支持 socks5/http/https，例如 socks5://user:pass@192.168.1.1:1080/
proxy-url: ""

# 为 true 时，无前缀模型请求只会匹配无前缀凭据（除非前缀与模型名相同）。
force-model-prefix: false

# 请求重试次数；当响应码为 403/408/500/502/503/504 时重试。
request-retry: 3

# 冷却中的凭据等待的最长时间（秒），超过则触发重试。
max-retry-interval: 30

# 配额超限时的处理
quota-exceeded:
  switch-project: true # 配额超限时是否自动切换其他项目
  switch-preview-model: true # 配额超限时是否自动切换预览模型

# 多凭据匹配时的路由策略
routing:
  strategy: "round-robin" # 轮询（默认）或 fill-first

# 是否为 WebSocket API (/v1/ws) 启用认证
ws-auth: false

# 当 > 0 时，为非流式响应每隔 N 秒发送空行以防止空闲超时
nonstream-keepalive-interval: 0

# 当为 true 时，为 Codex API 请求启用官方 Codex 指令注入
# 当为 false（默认）时，CodexInstructionsForModel 立即返回而不修改
codex-instructions-enabled: false

# 流式传输行为（SSE keep-alive 与安全启动重试）
streaming:
  keepalive-seconds: 15   # 默认：0（禁用）；≤0 关闭 keep-alive。
  bootstrap-retries: 1    # 默认：0（禁用）；首字节前的重试次数。

# Gemini API 密钥
gemini-api-key:
  - api-key: "AIzaSy...01"
    prefix: "test" # 可选：需要以 "test/gemini-3-pro-preview" 访问
    base-url: "https://generativelanguage.googleapis.com"
    headers:
      X-Custom-Header: "custom-value"
    proxy-url: "socks5://proxy.example.com:1080"
    models:
      - name: "gemini-2.5-flash" # 上游模型名
        alias: "gemini-flash"    # 客户端别名
    excluded-models:
      - "gemini-2.5-pro"     # 精确排除
      - "gemini-2.5-*"       # 前缀通配
      - "*-preview"          # 后缀通配
      - "*flash*"            # 子串通配
  - api-key: "AIzaSy...02"

# Codex API 密钥
codex-api-key:
  - api-key: "sk-atSM..."
    prefix: "test" # 可选：需以 "test/gpt-5-codex" 访问
    base-url: "https://www.example.com" # 自定义 Codex 端点
    headers:
      X-Custom-Header: "custom-value"
    proxy-url: "socks5://proxy.example.com:1080" # 可选：单独代理
    models:
      - name: "gpt-5-codex"   # 上游模型名
        alias: "codex-latest" # 客户端别名
    excluded-models:
      - "gpt-5.1"         # 精确排除
      - "gpt-5-*"         # 前缀通配
      - "*-mini"          # 后缀通配
      - "*codex*"         # 子串通配

# Claude API 密钥
claude-api-key:
  - api-key: "sk-atSM..." # 使用官方 Claude API 时无需 base-url
  - api-key: "sk-atSM..."
    prefix: "test" # 可选：需以 "test/claude-sonnet-latest" 访问
    base-url: "https://www.example.com" # 自定义 Claude 端点
    headers:
      X-Custom-Header: "custom-value"
    proxy-url: "socks5://proxy.example.com:1080" # 可选：单独代理
    models:
      - name: "claude-3-5-sonnet-20241022" # 上游模型名
        alias: "claude-sonnet-latest"      # 客户端别名
    excluded-models:
      - "claude-opus-4-5-20251101" # 精确排除
      - "claude-3-*"               # 前缀通配
      - "*-thinking"               # 后缀通配
      - "*haiku*"                  # 子串通配
    cloak:                         # 可选：为非 Claude Code 客户端进行请求伪装
      mode: "auto"                 # "auto"（默认）：仅当客户端不是 Claude Code 时伪装
                                   # "always"：始终应用伪装
                                   # "never"：从不应用伪装
      strict-mode: false           # false（默认）：将 Claude Code 提示前置到用户系统消息
                                   # true：删除所有用户系统消息，仅保留 Claude Code 提示
      sensitive-words:             # 可选：用零宽字符混淆的词汇
        - "API"
        - "proxy"

# OpenAI 兼容提供商
openai-compatibility:
  - name: "openrouter" # 提供商名称，用于 UA 等
    prefix: "test" # 可选：需以 "test/kimi-k2" 访问
    base-url: "https://openrouter.ai/api/v1" # 提供商基础 URL
    headers:
      X-Custom-Header: "custom-value"
    api-key-entries:
      - api-key: "sk-or-v1-...b780"
        proxy-url: "socks5://proxy.example.com:1080" # 可选：单独代理
      - api-key: "sk-or-v1-...b781" # 无代理
    models: # 提供商支持的模型
      - name: "moonshotai/kimi-k2:free" # 上游模型名
        alias: "kimi-k2" # 客户端别名

# Vertex API 密钥（Vertex 兼容端点，使用 API key + base URL）
vertex-api-key:
  - api-key: "vk-123..."                        # x-goog-api-key 头
    prefix: "test"                              # 可选前缀
    base-url: "https://example.com/api"         # 例如 https://zenmux.ai/api
    proxy-url: "socks5://proxy.example.com:1080" # 可选单独代理
    headers:
      X-Custom-Header: "custom-value"
    models:                                     # 可选：别名到上游模型
      - name: "gemini-2.5-flash"                # 上游模型名
        alias: "vertex-flash"                   # 客户端别名
      - name: "gemini-2.5-pro"
        alias: "vertex-pro"

# Amp 集成
ampcode:
  # Amp CLI OAuth 与管理功能的上游地址
  upstream-url: "https://ampcode.com"
  # 可选：覆盖 Amp 上游 API Key（否则使用环境变量或文件）
  upstream-api-key: ""
  # 按客户端的上游 API Key 映射
  # 将顶层 api-keys 中的客户端密钥映射到不同的 Amp 上游密钥。
  # 若未匹配到则回退到 upstream-api-key。
  upstream-api-keys:
    - upstream-api-key: "amp_key_for_team_a"    # 供这些客户端使用的上游密钥
      api-keys:                                 # 使用该上游密钥的客户端密钥
        - "your-api-key-1"
        - "your-api-key-2"
    - upstream-api-key: "amp_key_for_team_b"
      api-keys:
        - "your-api-key-3"
  # 是否将 Amp 管理路由 (/api/auth, /api/user 等) 仅限 localhost（默认 false）
  restrict-management-to-localhost: false
  # 是否在检查本地 API 密钥前强制执行模型映射（默认 false）
  force-model-mappings: false
  # Amp 模型映射：当请求的模型不可用时路由到本地可用模型
  # 适用于 Amp CLI 请求不可用模型（如 Claude Opus 4.5）但本地有相似模型的情况
  model-mappings:
    - from: "claude-opus-4-5-20251101"          # Amp 请求的模型
      to: "gemini-claude-opus-4-5-thinking"     # 路由到的可用模型
    - from: "claude-sonnet-4-5-20250929"
      to: "gemini-claude-sonnet-4-5-thinking"
    - from: "claude-haiku-4-5-20251001"
      to: "gemini-2.5-flash"

# 全局 OAuth 模型名称别名（按渠道）
# 这些别名同时用于模型列表和请求路由的模型 ID 重命名。
# 支持的渠道：gemini-cli、vertex、aistudio、antigravity、claude、codex、qwen、iflow。
# 注意：别名不适用于 gemini-api-key、codex-api-key、claude-api-key、openai-compatibility、vertex-api-key 或 ampcode。
# 您可以使用不同的别名重复相同的名称，以暴露多个客户端模型名称。
oauth-model-alias:
  antigravity:
    - name: "rev19-uic3-1p"
      alias: "gemini-2.5-computer-use-preview-10-2025"
    - name: "gemini-3-pro-image"
      alias: "gemini-3-pro-image-preview"
    - name: "gemini-3-pro-high"
      alias: "gemini-3-pro-preview"
    - name: "gemini-3-flash"
      alias: "gemini-3-flash-preview"
    - name: "claude-sonnet-4-5"
      alias: "gemini-claude-sonnet-4-5"
    - name: "claude-sonnet-4-5-thinking"
      alias: "gemini-claude-sonnet-4-5-thinking"
    - name: "claude-opus-4-5-thinking"
      alias: "gemini-claude-opus-4-5-thinking"
#   gemini-cli:
#     - name: "gemini-2.5-pro"          # 该渠道下的原始模型名
#       alias: "g2.5p"                  # 客户端可见别名
#       fork: true                      # 为 true 时保留原名并同时增加别名作为额外模型（默认：false）
#   vertex:
#     - name: "gemini-2.5-pro"
#       alias: "g2.5p"
#   aistudio:
#     - name: "gemini-2.5-pro"
#       alias: "g2.5p"
#   claude:
#     - name: "claude-sonnet-4-5-20250929"
#       alias: "cs4.5"
#   codex:
#     - name: "gpt-5"
#       alias: "g5"
#   qwen:
#     - name: "qwen3-coder-plus"
#       alias: "qwen-plus"
#   iflow:
#     - name: "glm-4.7"
#       alias: "glm-god"

# OAuth 提供商的模型排除列表
oauth-excluded-models:
  gemini-cli:
    - "gemini-2.5-pro"     # 精确排除
    - "gemini-2.5-*"       # 前缀通配
    - "*-preview"          # 后缀通配
    - "*flash*"            # 子串通配
  vertex:
    - "gemini-3-pro-preview"
  aistudio:
    - "gemini-3-pro-preview"
  antigravity:
    - "gemini-3-pro-preview"
  claude:
    - "claude-3-5-haiku-20241022"
  codex:
    - "gpt-5-codex-mini"
  qwen:
    - "vision-model"
  iflow:
    - "tstars2.0"

# 可选的 payload 配置
payload:
  default: # 默认规则仅在 payload 中缺少参数时设置
    - models:
        - name: "gemini-2.5-pro" # 支持通配符（如 "gemini-*"）
          protocol: "gemini" # 将规则限制为特定协议，选项：openai、gemini、claude、codex、antigravity
      params: # JSON 路径（gjson/sjson 语法）-> 值
        "generationConfig.thinkingConfig.thinkingBudget": 32768
  default-raw: # 默认原始规则在缺少时使用原始 JSON 设置参数（必须是有效的 JSON）
    - models:
        - name: "gemini-2.5-pro" # 支持通配符（如 "gemini-*"）
          protocol: "gemini" # 将规则限制为特定协议，选项：openai、gemini、claude、codex、antigravity
      params: # JSON 路径（gjson/sjson 语法）-> 原始 JSON 值（字符串按原样使用，必须是有效的 JSON）
        "generationConfig.responseJsonSchema": "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
  override: # 覆盖规则总是设置参数，覆盖任何现有值
    - models:
        - name: "gpt-*" # 支持通配符（如 "gpt-*"）
          protocol: "codex" # 将规则限制为特定协议，选项：openai、gemini、claude、codex、antigravity
      params: # JSON 路径（gjson/sjson 语法）-> 值
        "reasoning.effort": "high"
  override-raw: # 覆盖原始规则总是使用原始 JSON 设置参数（必须是有效的 JSON）
    - models:
        - name: "gpt-*" # 支持通配符（如 "gpt-*"）
          protocol: "codex" # 将规则限制为特定协议，选项：openai、gemini、claude、codex、antigravity
      params: # JSON 路径（gjson/sjson 语法）-> 原始 JSON 值（字符串按原样使用，必须是有效的 JSON）
        "response_format": "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"answer\",\"schema\":{\"type\":\"object\"}}}"
  filter: # 过滤规则从 payload 中删除指定的参数
    - models:
        - name: "gemini-2.5-pro" # 支持通配符（如 "gemini-*"）
          protocol: "gemini" # 将规则限制为特定协议，选项：openai、gemini、claude、codex、antigravity
      params: # 要从 payload 中删除的 JSON 路径（gjson/sjson 语法）
        - "generationConfig.thinkingConfig.thinkingBudget"
        - "generationConfig.responseJsonSchema"
          
          
```

到这里配置就可以暂时结束了。

>如果觉得配置文件太复杂，可以就把 host 设为 127.0.0.1，其他暂时不用管。

### 1.3 反代配置

这里我用 Codex Oauth 反代为例，演示一下如何操作。

![image.png](http://img.xilyfe.top/img/20260330195920583.png)

1. 点击登录后，应该会跳出下面的授权链接：
![image.png](http://img.xilyfe.top/img/20260330200107100.png)
2. 点击打开链接，就会跳到 OpenAI 的授权界面：
![image.png](http://img.xilyfe.top/img/20260330200225165.png)
3. 授权完成 CPA 里面就会显示 “认证成功”，在认证文件里面就可以看到 Codex 的配置文件：
![image.png](http://img.xilyfe.top/img/20260330201246290.png)

{{< admonition type=warning title="过程中可能出现的问题">}} 
1. **Codex OAuth登录报错**：梯子需要打开 **Tun 模式** 还要打开 **全局模式**。
2. **Codex 无法登录（os 10013 error）**：第一种可能大概率是电脑上没下载 Codex，在电脑上下载 Codex 然后重启再运行。第二种可能是 1455 端口被其他程序占用了，把占用的进程 kill 了就好了。
3. **Codex 登录成功，但是报错回调 URL 提交失败: oauth flow is not pending**：大概率是梯子问题，参考 Q1。
{{< /admonition >}}

### 1.4 连接测试

这里我们用 Cherry Studio 测试一下是不是成功了：

![image.png](http://img.xilyfe.top/img/20260330203845709.png)

需要注意，如果跑在本地 API 地址需要填 `http://localhost:8317` 而不是 `http://127.0.0.1:8317`，要不然会报错连接失败。API 秘钥可以在 CLIProxyApi 的配置面板里面修改，默认是 `your_api_key-1`。然后可以用的模型可以在 **认证文件** 面板查看：

![image.png](http://img.xilyfe.top/img/20260330204230461.png)

经过测试可以成功输出了：

![image.png](http://img.xilyfe.top/img/20260330204332597.png)


## 2. CC-Switch

### 2.1 前言

CC-Switch 可以统一管理 Claude Code、Codex 与 Gemini CLI 的供应商配置、MCP 服务器、Skills 扩展和系统提示词。我们可以把刚刚配置好的 api 服务加入 CC-Switch，这样就能在 Claudecode 里面使用了。

### 2.2 下载安装

1. 点击下载链接→[传送门](https://github.com/farion1231/cc-switch/releases/latest)←，进入CC-Switch的Github Release页面
2. 鼠标滚动到最下方选择适合自己版本的安装包，windows系统推荐下载普通msi后缀的安装包进行安装
![image.png](http://img.xilyfe.top/img/20260330204644006.png)
3. 安装后运行CC-Switch主程序，界面如下：
![image.png](http://img.xilyfe.top/img/20260330204706708.png)

### 2.3 配置

按照下面配置即可：
- **供应商名称**：任意
- **API Key**：在 CLIProxyApi 里面可以配置，默认是 `your-api-key-1`
- **请求地址**：按照公共本地部署的话填写 `http://localhost:8173`
- **API 格式**：OpenAI Chat Completions 
- **主模型**：可以用的模型可以在 CLIPorxyApi 里面的 **认证文件** 面板

### 2.4 在 ClaudeCode 里使用

![image.png](http://img.xilyfe.top/img/20260330205338177.png)

点击 switch model 后就能看到 custom model，也就是刚刚我们配置的 gpt-5.3-codex 了。