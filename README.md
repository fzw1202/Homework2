# 提示压缩

## 背景介绍

大规模语言模型在很多场景中已经展现出了强大的生成和推理能力。在不同的任务中，选择适合的提示（prompt）对于效果有着显著的影响。除了问题之外，提示中往往包含思维链（Chain-of-thought）、上下文学习示例（demonstration）、检索到的文档等。虽然这些方法能有效提高大规模语言模型在不同场景中的表现，但也使得提示变得更长，而更长的提示会要求更多的计算资源。因此，在巨大的计算需求和对更长提示的需求之间找到平衡变得迫在眉睫。

提示压缩[1,2]是一种针对以上问题的解决方法。该方法认为提示语句有内在的冗余性，部分文本对于语言模型是多余的，因而存在被压缩的可能性。一个随之而来的问题是：给定任意一个大规模语言模型，能否能设计一种自动提示压缩算法获得更短的提示以提高，但同时得到与原始结果相当、甚至更好的结果呢？

## 作业目标

- 面向简单数学问题的求解任务(GSM8K)，设计一种提示压缩算法对含有思维链的上下文学习提示进行压缩，使得大语言模型在压缩后的提示下仍然能保持一定性能。
- 自选一个语言模型，使用原有提示和压缩后的提示进行测试
    - 如果你能获取足够的GPU计算资源，不论是来自实验室服务器或者[Colab](https://colab.research.google.com/drive/)，那么你可以编写程序对大语言模型进行自动测试。我们提供了一份代码样例（仅供参考，不要求使用）。
    - 如果你无法获取计算资源，则可以手动通过一些在线平台手动输入提示进行测试，如[LMSYS](https://chat.lmsys.org/?arena)或[ChatGPT](https://chat.openai.com/)。
- 撰写一份分析文档（不超过2页），对比模型的性能变化，可额外分析提示压缩比例与模型性能的关系。


### 数据

`data.jsonl` 包含30条提示数据，每条数据有以下字段：
 - `prompt`: 原始提示，由几个示例和问题组成。
 - `label`: 最终答案，通常是一个数字，如`74`。
 - `answer`: 完整答案。除了最终答案外（位于末尾，被`####`分割），还有CoT分析过程。
 - `demo_type`: 原始提示中的上下文示例类型，共有三种：`complex`, `easy`, `mid`。来自[3]。




### 示例代码环境配置与要求 （可选）

- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
- CUDA version 12.1
- 安装Anaconda，并根据提供的`env.yml`创建虚拟环境（如果存在网络问题，请自行搜索配置清华大学镜像）

    ```bash
    conda env create --name hw2 --file env.yml
    conda activate hw2
    ```

- 配置huggingface镜像网站

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```


### 示例代码实现（可选）

- `main.py` 中已经实现了大部分的数据加载、模型加载、性能评测等功能实现。
- 请实现其中的`compress_prompt()`函数，对传入的`original_prompt`字符串进行压缩，返回一个`compressed_prompt`字符串。
- 运行测试：

    ```bash
    python3 main.py --model_name xxx --demo_type xxx
    ```
    -  参数设置
    - `model_name`: 用于评测性能的模型，我们建议使用`microsoft/phi-1_5` 或 `Qwen/Qwen1.5-0.5B`。也可以根据自己的计算资源选择探索其他支持的模型（见 [vllm supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)）
    - `demo_type`: 选取特定类型的提示，可选：`complex`, `easy`, `mid`, `all` (全集)。



## 提交内容


- 提示压缩的算法代码
- 最多两页的分析文档（PDF，格式不限）

请将分析文档（PDF版本）与压缩算法的代码文件统一打包为zip文件在教学网上提交，命名格式为：学号-第二次作业.zip，如2201213288-第二次作业.zip

## 说明

- 示例代码中使用了`vllm`库，对于GPU有较高要求。对于没有符合条件GPU计算资源的同学，可以参考示例代码自行实现程序，或直接使用线上平台评测（评分没有差异）。
- 数学问题的求解的性能以模型最终答案在数据集上的完全匹配准确率（Exact Match）衡量。本次作业提供的代码中，`evaluate_answers()`函数负责判断模型输出答案是否正确；如果不使用本次作业提供的代码，你可以人工从模型的输出中提取最终答案并进行判断。
- 计算压缩比时，可使用提示的字符数、单词数来衡量提示的长度。
- 不需要实现过于复杂的压缩方法。提示：1. 可尝试一些简单的方法，如删除部分示例、句子或单词。2. 可使用一些方法度量提示中不同内容的冗余度，再进行删除。3. 可使用摘要、复述等方式精简提示的表达。
- 参考文献中的方法仅供感兴趣的同学参考学习，不做强制要求。


## 截止日期

2024年4月21日24:00

## 参考文献

[1] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://aclanthology.org/2023.emnlp-main.825). In *EMNLP*, pages 13358–13376. Association for Computational Linguistics.

[2] Yucheng Li. 2023. [Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering](https://arxiv.org/abs/2304.12102).

[3] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, Tushar Khot. 2022. [Complexity-Based Prompting for Multi-step Reasoning](https://openreview.net/forum?id=yf1icZHC-l9). In *ICLR*. OpenReview.net.

[4] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. 2021. [Training verifiers to solve math word problems](https://arxiv.org/abs/2110.14168v2).

[5] Kaiyan Chang, Songcheng Xu, Chenglong Wang, et al. 2024. [Efficient Prompting Methods for Large Language Models: A Survey](https://arxiv.org/abs/2404.01077).