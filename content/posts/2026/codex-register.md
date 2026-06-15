---
title: Codex Free 无限续杯
date: 2026-06-10T13:30:00+08:00
authors:
  - Xilyfe
series:
  - 备忘
tags:
  - Cloudflare
  - Email
  - register
lastmod: 2026-06-10T11:37:23+08:00
featuredImage: http://img.xilyfe.top/img/20260610221816600.png
---
本来我真的是非常懒得折腾，直到我发现 CPA 里注册的九个 codex free 账号都用完了，token 焦虑又发作了不得不研究一下注册机。

![b6b80cf4-f604-4fba-9c58-8e2c27a2c156.png](http://img.xilyfe.top/img/20260610221656264.png)

>奥特曼你罪大恶极啊，codex 的额度从每周一刷新变成了每月一刷新。。。

## 整体流程

我们先来理一遍注册一个账号需要哪些步骤：
1. 首先注册账号肯定需要一个邮箱密码，但是邮箱从哪里来呢？如果用 126/163/gmail/hotmail 等大邮箱肯定可以，但是这些邮箱的批量注册都很麻烦。如果我们想用哪些临时邮箱，openai 会封锁这些账号。解决方案就是我们需要自己准备一个域名，例如 abc.com，然后我们就可以利用这个域名创建我们自己的邮箱，例如 `xxx@abc.com`。
2. 以前的 openai 可以用邮箱直接注册，但是最近奥特曼作妖导致新注册的账号需要绑定一个手机号，而且不能是大陆的 +86，这就导致我们需要用接码平台去找国外的手机号用。
3. 最后我们注册好账号该怎么使用呢？如果你看过这个系列的前一篇文章就应该知道，这时候该用 CPA 把这些账号收集到号池理反代。
4. 其次前面这三个流程都是很繁琐的，我们需要一个注册机替我们自动化的跑通这个流程。

## 部署临时邮箱

第一步我们需要准备一个域名，然后把域名托管到 Cloudflare，之后利用 github 项目接收邮件进行转发。

### 准备域名

1. 购买域名可以在腾讯云、阿里云或者能提供域名注册的云服务商购买。需要注意的一点是，购买的二级域名不能是 `.top`，建议是 `.xyz` 或者 `.cloud` 之类，`.top` 二级域名的邮箱会被 openai 屏蔽，应该是太多人薅羊毛了。

![image.png](http://img.xilyfe.top/img/20260610225726160.png)

2. 把域名绑定到 cloudflare，这一步在我上一盘文章讲过了，这里不在赘述，等到出现图片就说明成功了。

![image.png](http://img.xilyfe.top/img/20260610225828287.png)

### 部署环境

这一步我们需要部署 `cloudflare_temp_email` 这个项目。其实吧域名部署到 cloudflare 之后我们已经可以用这个邮箱了，但如果需要实现自动化注册，注册机就需要能通过 api 申请邮箱和读取邮件内容（验证码）。

1. 创建 D1 数据库，名字取 `mail_db` 即可。

![image.png](http://img.xilyfe.top/img/20260610230215951.png)

![image.png](http://img.xilyfe.top/img/20260610230230940.png)
2. 把 [db/schema.sql](https://github.com/dreamhunter2333/cloudflare_temp_email/blob/main/db/schema.sql) 里面 SQL 复制到控制台执行即可，执行完后概览里表数量是 10 就说明数据库初始化好了。

![image.png](http://img.xilyfe.top/img/20260610230411480.png)

![image.png](http://img.xilyfe.top/img/20260610230422044.png)

![image.png](http://img.xilyfe.top/img/20260610230431044.png)

3. 配置 KV 缓存，命名 `mail_kv`。

![image.png](http://img.xilyfe.top/img/20260610230455343.png)

![image.png](http://img.xilyfe.top/img/20260610230505911.png)
4. 现在数据库和 KV 缓存配置好了，需要配置项目的后端。创建 Workers 部署 `cloudflare_temp_email` 后端。

![image.png](http://img.xilyfe.top/img/20260610230610934.png)

![image.png](http://img.xilyfe.top/img/20260610230646206.png)

![image.png](http://img.xilyfe.top/img/20260610230658044.png)

5. 绑定DB数据库和KV缓存

![image.png](http://img.xilyfe.top/img/20260610230721272.png)

![image.png](http://img.xilyfe.top/img/20260610230729526.png)

![image.png](http://img.xilyfe.top/img/20260610230738532.png)

![image.png](http://img.xilyfe.top/img/20260610230753416.png)

![image.png](http://img.xilyfe.top/img/20260610230815450.png)

6. 配置变量参数，**注意参数类型**。

![image.png](http://img.xilyfe.top/img/20260610230846413.png)

7. 配置一下兼容性标志

![image.png](http://img.xilyfe.top/img/20260610230913456.png)

![image.png](http://img.xilyfe.top/img/20260610230934507.png)

>直接复制粘贴 nodejs_compat

8. 下载后端代码部署：[Releases · dreamhunter2333/cloudflare_temp_email · GitHub](https://github.com/dreamhunter2333/cloudflare_temp_email/releases)

![image.png](http://img.xilyfe.top/img/20260610231056663.png)

![image.png](http://img.xilyfe.top/img/20260610231103952.png)

![image.png](http://img.xilyfe.top/img/20260610231113064.png)

![image.png](http://img.xilyfe.top/img/20260610231123846.png)

>如果刷新后是 ok，恭喜你部署成功

9. 配置自定义域，改成 `apimail.你的域名`

![image.png](http://img.xilyfe.top/img/20260610231157357.png)

10. 重新配置域名电子路由

![image.png](http://img.xilyfe.top/img/20260610231232956.png)

![image.png](http://img.xilyfe.top/img/20260610231300971.png)


![image.png](http://img.xilyfe.top/img/20260610231248064.png)

![image.png](http://img.xilyfe.top/img/20260610231318776.png)

10. 部署前端页面。首先在 [Cloudflare Pages 前端 | 临时邮箱文档](https://temp-mail-docs.awsl.uk/zh/guide/ui/pages) 填入你的后端域名，刚刚那个 `apimail.你的域名`。

![image.png](http://img.xilyfe.top/img/20260610231435996.png)

![image.png](http://img.xilyfe.top/img/20260610231449733.png)

![image.png](http://img.xilyfe.top/img/20260610231458068.png)

![image.png](http://img.xilyfe.top/img/20260610231513848.png)

![image.png](http://img.xilyfe.top/img/20260610231522451.png)

11. 添加自定义域，这次可以命名为 `mail.你的域名`

![image.png](http://img.xilyfe.top/img/20260610231548778.png)

这时候应该就部署成功了，打开 `mail.域名` 就可以进入临时邮箱网站。

![image.png](http://img.xilyfe.top/img/20260610231741071.png)

## 接码平台

接下来就可以进入二阶段了，接码平台还是非常多了，我选的是最主流的 Hero SMS。

![image.png](http://img.xilyfe.top/img/20260610232015808.png)

这一步其实没什么多说的，注册账号，冲一点刀乐。值得一提的是，我们接码的手机号应该选择哪一个地区的？在文章发布的时候，英国的 +16 解码成功率最高，价格在 0.03$/号，也就是 2 分钱一个号还是比较划算的。

>这里的成功率指很多方面：
>1. 填写手机号之后，可以通过接收短信注册，不用 Whatsapp
>2. 手机号没有被 ban，很多地区的手机号可能已经被屏蔽了
>3. 手机号可能接收不到二维码

其次选择接码地区也可以参考下面这个网站 [OPENAI (ChatGPT) 短信价格对比](https://sms.fur.li/)

![image.png](http://img.xilyfe.top/img/20260610232416037.png)

## 注册机

现在万事俱备只欠东风，我们运行注册机就可以。这里我用的是 [FlowPilot](https://github.com/QLHazyCoder/FlowPilot) 这个项目，他是一个浏览器插件控制浏览器自动化注册。

>主要还是因为 linuxdo 太久没发言了，降到 2 级看不到那些牛逼的项目。

在 chrome/edge 里面安装插件就能看到下面如图的界面，我依次讲一下各个参数怎么填。

![image.png](http://img.xilyfe.top/img/20260610233005563.png)

1. CPA 如果部署在本地，地址填 `http://localhost:8317/management.html`，如果在服务器就填 `http://ip:port/management.html`。
2. 管理秘钥就是 CPA 登录的密码，如果忘了在设置里也能改。
3. 账号密码留空就可以，注册机会自动生成。
4. 邮箱服务和邮箱生成都用 Cloudflare Temp Email，清 cookies 打开
5. Cloudflare Temp Email 的 TEMP API 填 cloudflare worker 的地址，比如我的是 `https://mailapi.xilyfe.workers.dev`
6. Cloudflare Temp Email 的 ADMIN AUTH 就是邮箱服务管理员的登录密码
7. TEMP 域名点一下更新就出来了，就是你的域名。

![image.png](http://img.xilyfe.top/img/20260610233427623.png)

8. 注册方式选**手机号注册**，实测成功率比先邮箱注册在绑定手机高。主要原因是注册失败基本是接码的问题，先接码不会浪费时间。
9. 接码服务商选 HeroSMS，国家安装上面说的选择，美国的成功率肯定最高不过价格贵。
10. 其他设置都不怎么需要改，价格上限可以适当调整，短信等待时间可以改成 25 默认是 60。


接下来在 chrome 的隐私模式打开插件就可以运行了，目前的插件在运行中还会预计一些莫名其妙的问题，有时候需要手动辅助。