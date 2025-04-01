import json
import os
import random
import re
from Levenshtein import ratio
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"
acc_format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
iou_format_pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"


format_reward_factor = float(os.getenv("FORMAT_REWARD_FACTOR", 0.1))


def format_reward(content, reward_func):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count1 = content.count("<think>")
    think_count2 = content.count("</think>")
    answer_count1 = content.count("<answer>")
    answer_count2 = content.count("</answer>")
    if think_count1 == 1 and think_count2 == 1 and answer_count1 == 1 and answer_count2 == 1:
        if reward_func == "acc":
            return bool(re.match(acc_format_pattern, content, re.DOTALL))
        elif reward_func == "iou":
            return bool(re.match(iou_format_pattern, content, re.DOTALL))
    return 0

def iou_reward(content, sol):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    
    reward = 0.0
    # Try symbolic verification first
    try:
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth = ground_truth.replace("$","")
        ground_truth = eval(ground_truth)
            
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            bbox_match = re.search(bbox_pattern, content_answer)
            if bbox_match:
                bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                if iou(bbox, ground_truth) > 0.5:
                    reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
                
    return reward

def process_expression(s):
    # 使用正则表达式移除所有运算符（=+、-、*、/）周围的空格
    return re.sub(r'\s*([=+\-*/])\s*', r'\1', s)


def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?![A-Z])', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None

def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def default_accuracy_reward(content, sol):
    reward = 0.0
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(content)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    if reward == 0.0:
        gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                threshold = float(verify(answer_parsed, gold_parsed))
                if threshold > 0:
                    reward = 1.0
            except Exception:
                pass
                
    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_matches[-1].strip() if content_matches else content.strip()
            
            if ground_truth[0] == "$" and ground_truth[-1] == "$":
                ground_truth = ground_truth[1:-1]
            if student_answer[0] == "$" and student_answer[-1] == "$":
                student_answer = student_answer.replace("$","")
            student_answer = process_expression(student_answer)
            ground_truth = process_expression(ground_truth)

            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 1.0
            
            if reward == 0.0:
                # Check if ground truth contains numbers
                has_numbers = bool(re.search(r'\d', ground_truth))
                # Check if it's a multiple choice question
                has_choices = extract_choice(ground_truth)
                
                if has_numbers:
                    # For numeric answers, use exact matching
                    reward = numeric_reward(student_answer, ground_truth)
                    if reward is None:
                        reward = ratio(clean_text(student_answer), clean_text(ground_truth))
                    
                elif has_choices:
                    # For multiple choice, extract and compare choices
                    correct_choice = has_choices.upper()
                    student_choice = extract_choice(student_answer)
                    if student_choice:
                        reward = 1.0 if student_choice == correct_choice else 0.0
                else:
                    # For text answers, use fuzzy matching
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            
            if reward <= 0.6:
                reward = 0.0
            elif reward > 0.6 and reward < 1.0:
                reward = 0.1 
                
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward if reward else 0.0


def acc_reward(content, sol, reward_func):
    if reward_func == "acc":
        return default_accuracy_reward(content, sol)
    elif reward_func == "iou":
        return iou_reward(content, sol)
    return 0


def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    reward_func = extra_info["reward_func"]
    acc_score = acc_reward(solution_str, ground_truth, reward_func)
    format_score = format_reward(solution_str, reward_func)
    return acc_score + format_reward_factor*format_score