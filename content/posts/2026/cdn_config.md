---
title: CDN 加速博客和图床
date: 2026-01-15T11:35:30+08:00
featuredImage: http://img.xilyfe.top/img/20260115195745823.png
authors:
  - Xilyfe
series:
  - 备忘
tags:
  - CDN
lastmod: 2026-01-16T11:15:20+08:00
---
把博客部署到 Github Pages 确实很方便，不过由于种种原因在内地没法直连，于是研究用 Cloudflare 加速一下。

{{< admonition type=question title="Cloudflare 如何加速 GitHub Pages">}} 
首先，我们在域名提供商(例如我用的是阿里云的域名)将域名的 Nameserver 指向 Cloudflare：
- ns1.cloudflare.com
- ns2.cloudflare.com
这样 Cloudflare 就能接管域名的 DNS服务。当我们访问域名 `xilyfe.top` 时候，浏览器会请求 Cloudflare 的 DNS，得到最近节点 IP，然后访问最近 Cloudflare 边缘节点，而不是直接访问 Github Pages。而 Cloudflare 节点会代替我们访问 GitHub Pages 源站。
{{< /admonition >}}

## 加速 Github Page

### 配置 cloudflare

1. 进入[https://www.cloudflare-cn.com/](https://www.cloudflare-cn.com/)，注册账号并登录
2. 添加自己的域名

![image.png](http://img.xilyfe.top/img/20260116103522352.png)
3. 注意不需要带 https 或者 www、

![image.png](http://img.xilyfe.top/img/20260116103623473.png)
4. 选择 free 套餐

![image.png](http://img.xilyfe.top/img/20260116103712458.png)
5. 添加 DNS 记录

![image.png](http://img.xilyfe.top/img/20260116103853141.png)
一共需要添加五条记录如下：
```
1. A  @ 185.199.108.153
2. A  @ 185.199.109.153
3. A  @ 185.199.110.153
4. A  @ 185.199.111.153
5. CNAME @ xxxx.github.io
```

6. 修改域名的 NameServer，我这里以阿里云为例：

![image.png](http://img.xilyfe.top/img/20260116105432188.png)

![image.png](http://img.xilyfe.top/img/20260116105554593.png)

7. 返回 Cloudflare，如果出现下面标识说明已经成功了，DNS 还需要一段时间才能生效。

![image.png](http://img.xilyfe.top/img/20260116105907445.png)

### 配置 Github Repo

1. 进入 github.io 对应的仓库，进入 Settings：

![image.png](http://img.xilyfe.top/img/20260116110307110.png)

2. 左栏中的 **pages**，在  **Custom domain** 中输入自己的域名保存，如果成功会显示下图：

![image.png](http://img.xilyfe.top/img/20260116110155120.png)

## 加速图床 

### 配置 cloudflare

1. 回到 Cloudflare 面板，打开 **Workds and Pages** 这个 tab，点击**创建应用程序**：

![image.png](http://img.xilyfe.top/img/20260116110759495.png)

2. 选择 **从 Hello World! 开始**：

![image.png](http://img.xilyfe.top/img/20260116110836160.png)

3. 点击 **编辑代码**：

![image.png](http://img.xilyfe.top/img/20260116111003718.png)

4. 把代码复制进去，并且填写 Github Img Repo 的一些信息：

```js
// Website you intended to retrieve for users.
const upstream = "raw.githubusercontent.com";

// Custom pathname for the upstream website.
// (1) 填写代理的路径，格式为 /<用户>/<仓库名>/<分支>
const upstream_path = "****";

// github personal access token.
// (2) 填写github令牌
const github_token = "****";

// Website you intended to retrieve for users using mobile devices.
const upstream_mobile = upstream;

// Countries and regions where you wish to suspend your service.
const blocked_region = [];

// IP addresses which you wish to block from using your service.
const blocked_ip_address = ["0.0.0.0", "127.0.0.1"];

// Whether to use HTTPS protocol for upstream address.
const https = true;

// Whether to disable cache.
const disable_cache = false;

// Replace texts.
const replace_dict = {
  $upstream: "$custom_domain",
};

addEventListener("fetch", (event) => {
  event.respondWith(fetchAndApply(event.request));
});

async function fetchAndApply(request) {
  const region = request.headers.get("cf-ipcountry")?.toUpperCase();
  const ip_address = request.headers.get("cf-connecting-ip");
  const user_agent = request.headers.get("user-agent");

  let response = null;
  let url = new URL(request.url);
  let url_hostname = url.hostname;

  if (https == true) {
    url.protocol = "https:";
  } else {
    url.protocol = "http:";
  }

  if (await device_status(user_agent)) {
    var upstream_domain = upstream;
  } else {
    var upstream_domain = upstream_mobile;
  }

  url.host = upstream_domain;
  if (url.pathname == "/") {
    url.pathname = upstream_path;
  } else {
    url.pathname = upstream_path + url.pathname;
  }

  if (blocked_region.includes(region)) {
    response = new Response(
      "Access denied: WorkersProxy is not available in your region yet.",
      {
        status: 403,
      }
    );
  } else if (blocked_ip_address.includes(ip_address)) {
    response = new Response(
      "Access denied: Your IP address is blocked by WorkersProxy.",
      {
        status: 403,
      }
    );
  } else {
    let method = request.method;
    let request_headers = request.headers;
    let new_request_headers = new Headers(request_headers);

    new_request_headers.set("Host", upstream_domain);
    new_request_headers.set("Referer", url.protocol + "//" + url_hostname);
    new_request_headers.set("Authorization", "token " + github_token);

    let original_response = await fetch(url.href, {
      method: method,
      headers: new_request_headers,
      body: request.body,
    });

    let connection_upgrade = new_request_headers.get("Upgrade");
    if (connection_upgrade && connection_upgrade.toLowerCase() == "websocket") {
      return original_response;
    }

    let original_response_clone = original_response.clone();
    let original_text = null;
    let response_headers = original_response.headers;
    let new_response_headers = new Headers(response_headers);
    let status = original_response.status;

    if (disable_cache) {
      new_response_headers.set("Cache-Control", "no-store");
    } else {
      new_response_headers.set("Cache-Control", "max-age=43200000");
    }

    new_response_headers.set("access-control-allow-origin", "*");
    new_response_headers.set("access-control-allow-credentials", "true");
    new_response_headers.delete("content-security-policy");
    new_response_headers.delete("content-security-policy-report-only");
    new_response_headers.delete("clear-site-data");

    if (new_response_headers.get("x-pjax-url")) {
      new_response_headers.set(
        "x-pjax-url",
        response_headers
          .get("x-pjax-url")
          .replace("//" + upstream_domain, "//" + url_hostname)
      );
    }

    const content_type = new_response_headers.get("content-type");
    if (
      content_type != null &&
      content_type.includes("text/html") &&
      content_type.includes("UTF-8")
    ) {
      original_text = await replace_response_text(
        original_response_clone,
        upstream_domain,
        url_hostname
      );
    } else {
      original_text = original_response_clone.body;
    }

    response = new Response(original_text, {
      status,
      headers: new_response_headers,
    });
  }
  return response;
}

async function replace_response_text(response, upstream_domain, host_name) {
  let text = await response.text();

  var i, j;
  for (i in replace_dict) {
    j = replace_dict[i];
    if (i == "$upstream") {
      i = upstream_domain;
    } else if (i == "$custom_domain") {
      i = host_name;
    }

    if (j == "$upstream") {
      j = upstream_domain;
    } else if (j == "$custom_domain") {
      j = host_name;
    }

    let re = new RegExp(i, "g");
    text = text.replace(re, j);
  }
  return text;
}

async function device_status(user_agent_info) {
  var agents = [
    "Android",
    "iPhone",
    "SymbianOS",
    "Windows Phone",
    "iPad",
    "iPod",
  ];
  var flag = true;
  for (var v = 0; v < agents.length; v++) {
    if (user_agent_info.indexOf(agents[v]) > 0) {
      flag = false;
      break;
    }
  }
  return flag;
}
```

5. 点击右上角的 **部署**：

![image.png](http://img.xilyfe.top/img/20260116111156410.png)

6. 打开设置，添加 **自定义域**：

![image.png](http://img.xilyfe.top/img/20260116111315542.png)

7. 域的名称可以是 **任意前缀+域名**，例如：img.xilyfe.top：

![image.png](http://img.xilyfe.top/img/20260116111402241.png)
### 配置 picgo

在 picgo 设置中填写自定义域名就可以了

![image.png](http://img.xilyfe.top/img/20260116111506723.png)

