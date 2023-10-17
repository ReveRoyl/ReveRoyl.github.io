---
title: ChatGpt使用心得
top: false
cover: false
toc: true
mathjax: true
date: 2023-03-20 14:05
password:
summary:
tags:
- tips
categories:
- daily life
---

Chatgpt的对话框中，无法直接给我们输出图片。这个时候，我们可以借助Unsplash API，使得Chatgpt直接在对话的聊天框中输出图片：

> Unsplash API 是一个基于 REST 的 API，它提供了丰富的图像数据和功能。在这里，通过使用 Unsplash API，这就可以让Chatgpt可以通过编程方式搜索、浏览和下载 Unsplash 平台上的图像，从而实现在聊天对话中的预览。

要让Chatgpt使用Unsplash API，我们可以使用如下命令：

“从现在起, 当你想发送一张照片时，请使用 Markdown ,并且 不要有反斜线, 不要用代码块。使用 Unsplash API ([https://source.unsplash.com/1280x720/?](https://link.zhihu.com/?target=https%3A//source.unsplash.com/1280x720/%3F) < PUT YOUR QUERY HERE >)。如果你明白了，请回复“明白””
