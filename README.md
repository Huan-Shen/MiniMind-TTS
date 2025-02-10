# MiniMind-TTS
本项目是开源项目minimind的个人扩展，旨在从0开始训练一个小参数量的语音合成模型**MiniMind-TTS**

<div align="center">
</div>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/Huan-Shen/MiniMind-TTS?style=social)](https://github.com/Huan-Shen/MiniMind-TTS)
[![GitHub Code License](https://img.shields.io/github/license/Huan-Shen/MiniMind-TTS?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind-v)](https://github.com/jingyaogong/minimind-v/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/Huan-Shen/MiniMind-TTS/pulls)


</div>

<div align="center">
  <h3>"大道至简"</h3>
</div>

<div align="center">

中文

</div>

* 本开源项目旨在从0开始，训练一个小参数量的语音生成大模型**MiniMind-TTS**
* **MiniMind-TTS**同样极其轻量，最小版本体积仅为 GPT3 的约 $\frac{1}{7000}$，力求做到个人GPU也可快速推理甚至训练。
* 这不仅是一个开源模型的实现，也是语音大模型的教程。
* 这仅仅是我在minimid作者大佬的基础上自己改写的一个不那么美观的项目，借此检验一下自己近期的学习成果，也希望对您有所帮助。
  

  > 为防止误读，「从0开始」特指基于纯语言模型[MiniMind](https://github.com/jingyaogong/minimind)（这是一个完全从0训练的类GPT模型）做进一步的，从0到1的语音能力的拓展。
  > 若需详细了解后者，请参考minimind作者大佬的项目[MiniMind](https://github.com/jingyaogong/minimind)。


* 通过 **MiniMind-TTS**，本项目希望回答这些问题，帮助研究者在有限的硬件条件下理解语音合成大模型的核心原理。
* 目前，我仅仅完成了**MiniMind-TTS**的语音生成训练和推理的最初版本，有很多地方还需要打磨和优化，后续会逐步完善更多功能，敬请期待。


# 📌  Detail

MiniMind-TTS 的基座语言模型是MiniMind (LLM) [minimind](https://github.com/jingyaogong/minimind)，
具体的模型结构、训练细节、原理、测试效果等均可移步[minimind](https://github.com/jingyaogong/minimind)项目查阅。
此处为减少冗余，省略讨论LLM的相关部分，默认您已对MiniMind (LLM)的细节有基本的了解。


MiniMind-TTS的结构几乎不变，仅增加了对 text_token 和 speech_token 的结合部分。

此时，不妨思考2个很有意思的问题：什么叫做**L**arge **L**anguage **M**odel(LLM)？什么叫做多模态模型？

* 这里直接使用minimind作者的话：

  > 大语言模型（LLM）名字虽然带有语言二字，但它们其实与语言关系不大，这只是历史问题，更确切的名字应该是自回归 Transformer
  或者其他。
  LLM 更多是一种统计建模的通用技术，它们主要通过自回归 Transformer 来模拟 token 流，而这些 token
  可以代表文本、图片、音频、动作选择、甚至是分子等任何东西。
  因此，只要能将问题转化为模拟一系列离散 token 的流程，理论上都可以应用 LLM 来解决。
  实际上，随着大型语言模型技术栈的日益成熟，我们可能会看到越来越多的问题被纳入这种建模范式。也就是说，问题固定在使用 LLM
  进行『下一个 token 的预测』，只是每个领域中 token 的用途和含义有所不同。

* [李玺老师](https://person.zju.edu.cn/xilics#694283)的公开讲话同样佐证了类似观点（原话大意如下）：

  > 文本、视频、语音、动作等在人类看来属于「多模态」信号，但所谓的「模态」其实只是人类在信息存储方式上的一种分类概念。
  就像`.txt`和`.png`文件，虽然在视觉呈现和高级表现形式上有所不同，但它们本质上并没有根本区别。
  之所以出现「多模态」这个概念，仅仅是因为人类在不同的感知层面上对这些信号的分类需求。
  然而，对于机器来说，无论信号来自何种「模态」，最终它们都只是以一串二进制的「单模态」数字序列来呈现。
  机器并不会区分这些信号的模态来源，而只是处理和分析这些序列背后所承载的信息内容。

---

私以为，**G**enerative **P**retrained **T**ransformer (GPT) 比 **L**arge **L**anguage **M**odel (LLM)更为贴切，
因此本人表达上更习惯用"GPT"去代表LLM/VLM/类GPT架构的系列模型，而非为了蹭OpenAI的热度。

---

至此，我们可以用一句话总结GPT的所作所为：
GPT模型根据现有token预测输出下一个下下一个下下下一个token ...，直到模型输出结束符；此处的"token"其实并不需要一定是文本！

---

* 对于LLM模型，如果需要理解"图片"，我们只要把"图片"作为对一种特殊的从来没见过的"外国语言"，通过"外语词典"翻译后即可作为特殊的语言输入LLM
* 对于LLM模型，如果需要理解"音频"，我们只要把"音频"作为对一种特殊的从来没见过的"外国语言"，通过"外语词典"翻译后即可作为特殊的语言输入LLM
* ...

---

<u>**所以，为了得到MiniMind-TTS，我们只需要完成2件事即可：**</u>

1. 借助擅长翻译音频的的 **"外语词典"** ，把音频从 **"外国语言"** 翻译为模型便于理解的 **"LLM语言"**
2. 训练微调LLM，使其和 **"外语词典"** 度过磨合期，从而更好的理解音频

---

"外语词典"一般称之为音频量化模型。
和Encodec、SoundStream等模型类似，MiniMind-TTS同样选用开源Clip系列模型作为量化器。
具体使用[WavTokenizer](https://github.com/jishengpeng/WavTokenizer)，
他是单码本的量化器，使用体验自己感觉效果还可以，就选择了他，方便搭建网络。

代码中借鉴了CosyVoice中对于拼接 Text_token、speech_token 的代码内容，感谢大佬的分享。

# todo
* [x] 完成音频量化模型WavTokenizer的搭建
* [x] 完成MiniMind-TTS的搭建
* [x] 完成MiniMind-TTS的推理
* [x] 调通MiniMind-TTS的训练、损失能够稳定下降
* [ ] 训练一版底模，看一下最终的效果如何
* [ ] 完成MiniMind-TTS的优化

# License
This repository is licensed under the [Apache-2.0 License](LICENSE).
