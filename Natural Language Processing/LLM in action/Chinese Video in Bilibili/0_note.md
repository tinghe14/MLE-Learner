Reference Video Series: https://www.bilibili.com/video/BV1jk4y1L75w/?spm_id_from=333.337.search-card.all.click&vd_source=8b4794944ae27d265c752edb598636de
Objective: to build personal chatgpt

### LLMs 实践 01 llama、alpaca、vicuna 整体介绍及 llama 推理过程
- summary
- llama weights
- infernece/sample/generator: [example.py](https://github.com/facebookresearch/llama/blob/main/example.py)
#### summary
- llama => alpaca => vicuna
  - llama: pretrained model, 作用类似于text-davini-003 (gpt3)
  - self instruct (fine tune)
    - alpaca: promopt, answer from chatgpt
    - vicuna: prompt from shareGPT(全世界2万多用户产生的) answer from chatgpt 4, long conversation/context window
      - 两者都基于llama[预训练非常烧钱](https://lmsys.org/blog/2023-03-30-vicuna/#:~:text=Table%201.%20Comparison%20between%20several%20notable%20models)
