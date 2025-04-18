from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify
# from utils import default_accuracy_reward


# BSZ=50 # reduce it if GPU OOM
# Qwen2VLImageProcessor
MODEL_PATH_list = [
    
    "minglingfeng/Ocean_R1_3B_Instruct",
    "minglingfeng/Ocean_R1_7B_Instruct",

]


for MODEL_PATH in MODEL_PATH_list:
    print(MODEL_PATH)

    ckpt_name = "_".join(MODEL_PATH.split("/")[-5:])
    BSZ = 50 if "3B" in MODEL_PATH else 20
    OUTPUT_PATH=f"./eval/logs/results/geoqa_grpo_{ckpt_name}.json"
    PROMPT_PATH="./eval_data/prompts/geoqa_test_prompts.jsonl"

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )



    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"

    data = []
    with open(PROMPT_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))


    QUESTION_TEMPLATE =  "{Question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."

    messages = []

    data = data

    img_dir = "{geoqa img path}"

    for i in data:
        message = [
            
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{img_dir}{i['image_path'][1:]}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=i['question'])
                }
            ]
        }]
        messages.append(message)




    all_outputs = []  # List to store all answers

    # Process data in batches
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")





    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data,all_outputs):
        original_output = model_output
        ground_truth = input_example['ground_truth']
        model_answer = parse(original_output) 

        # Count correct answers
        if model_answer is not None and float(verify(model_answer,parse(ground_truth)))>0:
            correct_number += 1
            is_correct = True
        else:
            is_correct = False

        ## minglingfeng fix
        # reward, model_answer = default_accuracy_reward(original_output, ground_truth)
        # if reward == 1.0:
        #         correct_number += 1
        #         is_correct = True
        # else:
        #     is_correct = False 
        
        try:
            result = {
                'question': input_example,
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer':str(model_answer) if model_answer is not None else None,
                'is_correct':is_correct
            }

        except Exception as e:
            print("no answer parsed",e,model_answer)
            result = {
                'question': input_example,
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer':None,
                'is_correct':is_correct
            }



        final_output.append(result)


    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")





