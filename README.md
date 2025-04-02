# Ocean-R1: An Open and Generalizable Large Vision-Language Model enhanced by Reinforcement Learning

## 🎯Overview

<!-- Inspired by the robust reasoning capabilities demonstrated by [DeepSeek R1](https://arxiv.org/abs/2501.12948)  in the text domain, we seek to extend the large-scale reinforcement learning (RL) techniques that have proven effective for large language models (LLMs) to multimodal scenarios. -->

Current implementations of [DeepSeek R1](https://arxiv.org/abs/2501.12948) framework in multimodal settings have predominantly concentrated on unimodal task specialization (e.g., mathematical reasoning, visual-spatial analysis, referential comprehension, or quantitative visual parsing). This narrow focus induces catastrophic forgetting - where optimization for isolated task domains compromises model generalization through capacity competition in shared neural substrates. Furthermore, while existing reproductions prioritize base model architectures due to stability concerns during reinforcement learning (RL) exploration, the critical question of metacognitive emergence ("Aha Moments") in instruction-tuned multimodal systems like Qwen2.5-VL-Instruct remains unaddressed.

To address these limitations, we propose **Ocean-R1**, a two-stage rule-based RL framework for multimodal intelligence enhancement:

🔥We open-source our complete pipeline to foster further research in this area. We release all our codes, model, data. We apply the awesome [verl](https://github.com/volcengine/verl) framework to train our models. Thanks for their great work!
- [🤗 Ocean-R1-3B-Instruct](https://huggingface.co/minglingfeng/Ocean_R1_3B_Instruct)
- [🤗 Ocean-R1-7B-Instruct](https://huggingface.co/minglingfeng/Ocean_R1_3B_Instruct)

- [🤗 Ocean_R1_visual_data_stage1](https://huggingface.co/datasets/minglingfeng/Ocean_R1_visual_data_stage1)
- [🤗 Ocean_R1_visual_data_stage2](https://huggingface.co/datasets/minglingfeng/Ocean_R1_visual_data_stage2)

> [!NOTE] 
> These data are from the open source community and are obtained through cleaning and filtering.


---

### 🚀 News
- 2025-04-03: We release the latest [Ocean-R1 repo](https://github.com/VLM-RL/Ocean-R1), including codebase, model, and training datasets.
- 2025-03-10: We release the [Ocean-R1 repo](https://github.com/fengzi258/Ocean-R1), including codebase, model, and training datasets.

--- 

### 🗞️ Our Findings
<a id="examples"></a>
| Examples of Reasoning Behaviors on GeoQA| 
|----------|
| <span style="color: blue;">However, this calculation seems to have an error. Let's re-evaluate the problem.</span>| 
| <span style="color: blue;">However, upon re-evaluating the correct proportionality and area relationships</span> | 
| <span style="color: blue;">Since CP = CD - DP = 8 - 16 = -8, which is not possible, we need to re-evaluate the setup.</span> | 
| <span style="color: blue;">However, this calculation is incorrect. Let's re-evaluate: </span>| 
| Therefore, angle A = angle ADO. Given that angle D = 50°, we have angle A = angle ADO = 50°. Since DE is parallel to OA, we have angle AOD = angle A + angle ADO = 100°. Since CD is the diameter, we have angle C = 90° - angle AOD = 90° - 100° = -10°. <span style="color: blue;">However, this calculation is incorrect. Let's re-evaluate: </span>Since DE is parallel to OA, we have angle AOD = 2 * angle D = 100°. Therefore, angle C = 90° - angle AOD = 90° - 100° = -10°. <span style="color: blue;">This is still incorrect. Let's re-evaluate correctly: </span>| 
| <span style="color: blue;">However, this contradicts the given conditions. Therefore, we need to re-evaluate the correct proportion, which is \( \frac{AB}{AC} = \frac{AD}{AB} \). | 
| Therefore, \( PM = \frac{1}{2} AD \) and \(QN = \frac{1}{2} BC \). Given that \( MP + NQ = 14 \) and \( AC + BC = 18 \), we have \( \frac{1}{2} AC + \frac{1}{2} BC = 14 \). Thus, \( AC + BC = 28 \). <span style="color: blue;">However, we are given that \( AC + BC = 18 \), which seems to be a contradiction. Revising the given conditions, we use the correct relationship: | 
| <span style="color: blue;">However, this calculation seems to have an error. Let's recheck the proportion ... |
| <span style="color: blue;">However, this calculation is incorrect. The correct approach is to use the fact that ... | 
| <span style="color: blue;">However, this calculation is incorrect. The correct calculation should be: |

- **Visual Aha Moment**: Our findings indicate that in multimodal reasoning scenarios, rule-based reinforcement learning (RL) training of the Qwen2.5vl-3B-Instruct and Qwen2.5vl-7B-Instruct models can elicit emergent behaviors such as reflection and verification, which are akin to "Aha Moment". Table [1](#examples) presents specific instances of these reflective and verification behaviors.

- **The Training Dynamic of Reasoning Behaviors**: Figure [1](#training_curve) depicts the evolution of reasoning behavior counts during model training on the GeoQA dataset. Notably, distinct models exhibit divergent trends in reasoning behavior changes. Importantly, reflection and verification behaviors do not invariably result in correct outcomes, as illustrated in the cases outlined in Table [2](#cases).

- **Response Length Depends on the data Difficulty**: We observed that RL training based on the Qwen2.5-VL-3B-Instruct model significantly enhances both accuracy and response length. Conversely, for the Qwen2.5-VL-7B-Instruct model, while accuracy improves, response length diminishes. This dichotomy suggests an inverse correlation between model capacity and perceived task difficulty, where larger models may encode information more efficiently.
<!-- We observed that RL training based on the Qwen2.5-VL-3B-Instruct model significantly improved accuracy and response length, but on Qwen2.5-VL-7B-Instruct only accuracy improved while response length decreased. We suspect that the training data is more difficult for Qwen2.5-VL-3B-Instruct, but for Qwen2.5-VL-7B-Instruct, the difficulty is relatively low. -->

- **Cross-Task Knowledge Transfer via Compositional Reasoning**: Following the enhancement of complex visual reasoning capabilities (Geoqa +20.1/+22.22, MathVision +2.5/+2.7, MathVista +4.4/+4.5) in stage 1, we achieved substantial improvements in the model's performance on visual counting (+28.6/+22.6), referring expression comprehension (+12.0/-0.5), and visual spatial reasoning (+11.0/+6.5) tasks using merely 20k mixed samples. This suggests that augmenting complex visual reasoning abilities can positively transfer to simpler tasks.


<a id="training_curve"></a>
<!-- ![Figure 1](./assets/training_curve.png) -->
<img width="600" src=./assets/training_curve.png />

<a id="cases"></a>
|Image|Question|Ground Truth |Ocean-R1-Instruct (3B or 7B)|Verify|
|:----------:|:----------|:----------|:----------|:-----:|
|![](./assets/319.png)|As shown in the figure, the straight line BC is tangent to circle O at point A, AD is the chord of circle O. Connect OD, if angle DAC = 50.0, then the degree of angle ODA is () |40°|\<think>\nTo solve for the measure of angle ODA, we need to use some properties of circles and tangents.\n\n1. Since the line \\(BC\\) is tangent to the circle at point \\(A\\), the radius \\(OA\\) is perpendicular to \\(BC\\). Therefore, \\(\\angle OAB = 90^\\circ\\).\n\n2. Since \\(AD\\) is a chord and \\(OA\\) is a radius, \\(\\angle AOD = 2 \\times \\angle DAC\\) because the angle at the center is twice the angle at the circumference subtended by the same arc. Given that \\(\\angle DAC = 50^\\circ\\), we have:\n   \\[\n   \\angle AOD = 2 \\times \\angle DAC = 2 \\times 50^\\circ = 100^\\circ\n   \\]\n\n3. Since \\(OA \\perp BC\\), \\(\\angle OAB = 90^\\circ\\). Therefore, in \\(\\triangle OAD\\), we have:\n   \\[\n   \\angle ODA = 180^\\circ - \\angle OAD - \\angle AOD = 180^\\circ - 90^\\circ - 100^\\circ = 180^\\circ - 190^\\circ = 80^\\circ\n   \\]\n\nThus, the measure of angle \\(ODA\\) is \\(\\boxed{40^\\circ}\\).\n\n<span style="color: blue;">However, upon reviewing the options and the problem statement, it seems there might be a discrepancy.</span> Given the options, the correct measure of \\(\\angle ODA\\) should be derived from the properties of the circle and the given angles. Since \\(\\angle AOD = 100^\\circ\\) and knowing that \\(OA \\perp BC\\), the correct measure of \\(\\angle ODA\\) should be:\n\\[\n\\angle ODA = 90^\\circ - \\angle DAC = 90^\\circ - 50^\\circ = 40^\\circ\n\\]\n\nTherefore, the correct answer is \\(\\boxed{40^\\circ}\\).\n\</think>\n\<answer>\n40°\n\</answer>|YES|
|![](./assets/357.png)|As shown in the figure, in the parallelogram ABCD, F is a point on AB, DF intersects AC at point E, if CD = 10.0, frac {AE}{EC}=frac {2.0}{5.0}, then the length of BF is ()|6|\<think>\nSince quadrilateral ABCD is a parallelogram, we have AB = CD = 10. Also, AD parallel BC, which implies ∠DAE = ∠FCE and ∠ADE = ∠CFE. Therefore,△ADE ∼ △CFE. Thus, \\(\\frac{AE}{EC} = \\frac{AD}{FC}\\). Given \\(\\frac{AE}{EC} = \\frac{2}{5}\\), we have \\(\\frac{AD}{FC} = \\frac{2}{5}\\). Since AD = 10, we get \\(FC = \\frac{5 \\times 10}{2} = 25\\). Therefore, \\(BF = AB - AF = 10 - (25 - 10) = 10 - 15 = -5\\). <span style="color: blue;">However, this calculation seems to have an error. Let's recheck the proportion. </span>Since \\(\\frac{AE}{EC} = \\frac{2}{5}\\), we have \\(\\frac{AD}{FC} = \\frac{2}{5}\\). Therefore, \\(FC = \\frac{5 \\times 10}{2} = 25\\). Thus, \\(BF = AB - AF = 10 - (25 - 10) = 6\\). Therefore, the length of BF is 6.\n\</think>\n\n\<answer>\n6\n\</answer>|YES|
|![](./assets/608.png)|As shown in the figure, CD is the diameter of circle O, chord DE parallel  OA, if the degree of angle D is 50.0, then the degree of angle C is ()|25°|\<think>\nSince OA is the radius, we have OA = OD. Therefore, angle A = angle ADO. Given that angle D = 50°, we have angle A = angle ADO = 50°. Since DE is parallel to OA, we have angle AOD = angle A + angle ADO = 100°. Since CD is the diameter, we have angle C = 90° - angle AOD = 90° - 100° = -10°. <span style="color: blue;">However, this calculation is incorrect. Let's re-evaluate:</span> Since DE is parallel to OA, we have angle AOD = 2 * angle D = 100°. Therefore, angle C = 90° - angle AOD = 90° - 100° = -10°. <span style="color: blue;">This is still incorrect. Let's re-evaluate correctly: Since DE is parallel to OA, we have angle AOD = 2 * angle D = 100°.</span> Therefore, angle C = 90° - angle AOD = 90° - 100° = 20°. Thus, the degree of angle C is 20°.\n\</think>\n\n\<answer>\n20°\n\</answer>|NO|


|Model|Response Length|Score |val score 1|val score 2|
|:----------:|:----------|:----------|:----------|:-----:|
|Qwen2.5-VL-3B-Instruct-stage1|![](./assets/3B-stage1_response_length.png)|![](./assets/3B-stage1_score.png) |![](./assets/3B-stage1_geoqa.png)|![](./assets/3B-stage1_cvbench.png)|
|Qwen2.5-VL-3B-Instruct-stage2|![](./assets/3B-stage1_response_length.png)|![](./assets/3B-stage1_score.png) |![](./assets/3B-stage1_geoqa.png)|![](./assets/3B-stage1_cvbench.png)|
||||||
|Qwen2.5-VL-7B-Instruct-stage1|![](./assets/7B-stage1_response_length.png)|![](./assets/7B-stage1_score.png) |![](./assets/7B-stage1_val.png)|![](./assets/7B-stage1_cvbench.png)|
|Qwen2.5-VL-7B-Instruct-stage2|![](./assets/7B-stage2_response_length.png)|![](./assets/7B-stage2_score.png) |![](./assets/7B-stage2_val.png)|![](./assets/7B-stage2_cvbench.png)|


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
            "ability": "math", ## default
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'index': idx,
                'answer': answer,
                "question": problem,
                "reward_func": reward_func, ## acc or iou
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
- Upload Arxiv
- Synthesize more high-quality and diverse multimodal data
- Scale up to larger models and more general tasks

## 🤝 Acknowledgements

We sincerely thank [verl](https://github.com/volcengine/verl) (our initial codebase), [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR), [RefCOCO](https://github.com/lichengunc/refer), and [CVBench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) for providing open source resources and to build the project. 


## 📚 Contributors and Citation

Contributors: [Lingfeng Ming](https://scholar.google.com/citations?user=QOMvlswAAAAJ&hl=zh-CN), [Yadong Li](https://scholar.google.com/citations?user=VLfXcYIAAAAJ&hl=en), Song Chen, Jianhua Xu, Zenan Zhou, Weipeng Chen. 

If you find this work useful, please cite it as follows:
```bib
@misc{ming2025oceanr1,
  author       = {Lingfeng Ming, Yadong Li, Song Chen, Jianhua Xu, Zenan Zhou, Weipeng Chen},
  title        = {Ocean-R1: An Open and Generalizable Large Vision-Language Model enhanced by Reinforcement Learning},
  howpublished = {\url{https://github.com/VLM-RL/Ocean-R1}},
  note         = {Accessed: 2025-04-03},
  year         = {2025}
}
```