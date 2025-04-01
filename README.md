# Ocean-R1: An Open and Generalizable Large Vision-Language Model enhanced by Reinforcement Learning

## 🎯Overview

Inspired by the robust reasoning capabilities demonstrated by [DeepSeek R1](https://arxiv.org/abs/2501.12948)  in the text domain, we seek to extend the large-scale reinforcement learning (RL) techniques that have proven effective for large language models (LLMs) to multimodal scenarios.

We apply the awesome [verl](https://github.com/volcengine/verl) framework to train our models. Thanks for their great work!



---

### 🚀 News
- 2025-04-03: We release the latest [Ocean-R1 repo](https://github.com/VLM-RL/Ocean-R1), including codebase, model, and training datasets.
- 2025-03-10: We release the [Ocean-R1 repo](https://github.com/fengzi258/Ocean-R1), including codebase, model, and training datasets.

--- 

### 🗞️ Our Findings
- **R1's "Aha Moment" in Visual Reasoning on a 3B Instruct Model**:  

## 📦 Setup

```shell
git clone https://github.com/VLM-RL/Ocean-R1
cd Ocean-R1
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

> [!NOTE] 
> If you meet bug when running the script, first try align your environments with `requirements.txt`



## 🔄 Training

### Data Preparation
You can download our training data from [Ocean_R1_visual_data_stage1](https://huggingface.co/datasets/minglingfeng/Ocean_R1_visual_data_stage1) and [Ocean_R1_visual_data_stage2](https://huggingface.co/datasets/minglingfeng/Ocean_R1_visual_data_stage2). Each entry in our datasets is a dictionary organized in the following format. 
```json
data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "images": images,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'index': idx,
                'answer': answer,
                "question": problem,
                "reward_func": reward_func,
                "image_paths": image_paths
            }
        }
```

### Customized Reward Function
We implement customized reward functions in a separate file and specify them using `custom_reward_function.path` and `custom_reward_function.name`. Please refer to `./verl/verl/utils/reward_score/custom_reward_fn.py` for more details.


### Start Training (GRPO)
- for single node

  ```shell
  bassh ./verl/examples/grpo_trainer/run_qwen25vl-3b_stage1.sh
  bassh ./verl/examples/grpo_trainer/run_qwen25vl-3b_stage2.sh
  ```

- for multiple node

  ```shell
  bash ./verl/examples/grpo_trainer/run_qwen25vl-3b_multinodes_stage1.sh
  bash ./verl/examples/grpo_trainer/run_qwen25vl-3b_multinodes_stage2.sh
  ```

## 🧪 Evaluation
> [!NOTE] 
> The models are evaluated in the `zero-shot` setting and with an `extracted matching` approach, which corresponds to the rule-based reward in training stage. We provide the following evaluation scripts for reproduction.


| Model       | SuperCLEVR       |GEOQA       |RefCOCO/+/g AVG     |MathVision       |MathVerse       |OlympiadBench       |MMMU       |
|:-----------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Qwen2.5-VL-3B-Instruct   |64.1    |37.0    |75.3    |14.4    |27.6    |14.6    |40.5    |
| Qwen2.5-VL-3B-Instruct-GRPO-text   | 66.1   |38.7    |2.4    |17.4   |31.5    |14.8    |43.4    |
| Qwen2.5-VL-3B-Instruct-GRPO-vis   | 93.4   | 54.2   |86.1    |19.1    |40.0    |15.5    |47.9    |

### Visual Counting: SuperCLEVR

```bash
cd ./eval_data
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip

# change image dir and the model path in the scripts
python ./eval/test_qwen2vl_counting_superclevr_5k.py

```

### Geometric Reasoning: GEOQA

We provide the example script to evaluate on the test set (direct answer form) of [GEOQA](https://arxiv.org/abs/2312.11370).


```bash
# prepare images for testing
cd ./eval_data
git lfs install
git clone https://huggingface.co/datasets/Luckyjhg/Geo170K
cd Geo170K
unzip images.zip


# change image dir and the model path in the scripts
python ./eval/test_qwen2vl_geoqa.py

```

### Referring Expression Comprehension (REC): RefCOCO/+/g
> 1. Download the [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip) and unzip it, and we refer to the image dir as `<your_image_root>`.

> 2. Download the [RefCOCO/+/g Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) and unzip it.

```bash
# Remember to change the model path, image root, and annotation path in the script
python ./eval/test_qwen2d5vl_rec.py
```

### Visual Spatial Reasoning: CVBench
```bash
python ./eval/test_qwen2d5vl_cvbench.py
```

### Math: MathVision, MathVerse and MathVista
We apply [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate the math Benchmarks.

## 📋️ TODO
- Synthesize more high-quality and diverse multimodal data
- Scale up to larger models and more general tasks

## 🤝 Acknowledgements

We sincerely thank [verl](https://github.com/volcengine/verl) (our initial codebase), [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR), [RefCOCO](https://github.com/lichengunc/refer), and [CVBench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) for providing open source resources and to build the project. 


## 📚 Contributors and Citation

Contributors: [Lingfeng Ming](https://scholar.google.com/citations?user=QOMvlswAAAAJ&hl=zh-CN), Youwei Zhang, [Yadong Li](https://scholar.google.com/citations?user=VLfXcYIAAAAJ&hl=en), Song Chen, Jianhua Xu, Zenan Zhou, Weipeng Chen. 

If you find this work useful, please cite it as follows:
```bib
@misc{ming2025oceanr1,
  author       = {Lingfeng Ming, Youwei Zhang, Yadong Li, Song Chen, Jianhua Xu, Zenan Zhou, Weipeng Chen},
  title        = {Ocean-R1: An Open and Generalizable Large Vision-Language Model enhanced by Reinforcement Learning},
  howpublished = {\url{https://github.com/VLM-RL/Ocean-R1}},
  note         = {Accessed: 2025-04-03},
  year         = {2025}
}
```