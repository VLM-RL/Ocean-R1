import yaml
import json
import os
import random
from PIL import Image
import math
from torch.utils.data import Dataset

simple_system_prompt = "You are a helpful assistant. Please solve the question. The user asks a question, and you solves it."

general_system_prompt = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


# simple_question_template = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

simple_question_template = "{Question}\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."

general_question_template = """You should first think about the reasoning process in the mind and then provide the user with the answer. Your answer must be in latex format and wrapped in $...$. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think> <answer> $2$ </answer>, which means your output should start with <think>and end with </answer>.\n Question: {Question}"""

math_question_template = """You should first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>.\n"""

iou_question_template = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str):
        super(LazySupervisedDataset, self).__init__()
        
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        
        self._shuffle()
        print(f"Total Samples: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)
    
    def _shuffle(self):
        random.shuffle(self.list_data_dict)

    def write_to_jsonl(self, save_dir, file_name):
        with open(os.path.join(save_dir, file_name), "w", encoding="utf-8")as jsonl_file:
            for item in self.list_data_dict:
                json_line = json.dumps(item, ensure_ascii=False)
                jsonl_file.write(json_line + "\n")


    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": simple_system_prompt},
                    {"role": "user", "content": example["question"]},
                ],
            }
        
        def make_conversation_image(example):
            
            contents = [
                        {"type": "text", "text": iou_question_template.format(Question=example["question"]) if "reward_func" in example and example["reward_func"] == "iou" else simple_question_template.format(Question=example["question"])},
                    ]
            image_paths = example["image_paths"]
            for img_path in image_paths:
                contents.append(
                    {"type": "image", "image": f"file://{img_path}"}
                )
                    

            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": contents,
                    },
                ],
            }

        example = self.list_data_dict[i]

        return {
            'image_paths': example["image_paths"] if 'image_paths' in example else [] ,
            'problem': example['question'],
            'solution': example['answer'],
            "reward_func": example["reward_func"],
            'prompt': make_conversation_image(example)["prompt"] if 'image_paths' in example else make_conversation(example)["prompt"],
        }


if __name__ == "__main__":  

    # data_path = "/data_train2/mllm/minglingfeng/code/lmm-r1/examples/data/data_configs/visual_all_data_stage2.yaml"
    data_path = "/data_train2/mllm/minglingfeng/code/verl/examples/data_preprocess/data_configs/visual_all_data_stage2_v1.yaml"
    
    dataset = LazySupervisedDataset(data_path)

    save_dir = "/data_train2/mllm/minglingfeng/mllm_data/processed_data/r1_data/"
    file_name = data_path.split("/")[-1].split(".")[0] + "_verl.jsonl"
    dataset.write_to_jsonl(save_dir, file_name)
    print(f"Save to {file_name}")