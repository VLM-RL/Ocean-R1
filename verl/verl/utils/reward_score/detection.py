# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime

from datasets import load_dataset, load_from_disk, Image, Value, Features 

import numpy as np
from scipy.optimize import linear_sum_assignment
import json

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
        try:
            content_str = content_str.split("```json\n")[1].split("\n```")[0]
            bbox_list = json.loads(content_str)
        except:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def extract_pred_bbox(response):
    # 匹配所有符合格式的JSON对象字符串
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[[^\]]+\],\s*"label"\s*:\s*"[^"]+"\s*\}'
    
    # 使用非贪婪模式匹配，避免跨对象匹配
    matches = re.findall(pattern, response, re.DOTALL)
    
    # 解析所有匹配的JSON对象
    result = []
    for match in matches:
        try:
            # 清洗可能的尾部逗号
            cleaned = re.sub(r',\s*}', '}', match)
            obj = json.loads(cleaned)
            result.append(obj)
        except json.JSONDecodeError:
            # 处理特殊格式问题
            try:
                # 尝试修复引号问题
                fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
                obj = json.loads(fixed)
                result.append(obj)
            except:
                continue
    if len(result) == 0:
        result = None
    return result

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area + 1e-6
    
    iou = intersection_area / union_area
    return iou

def calculate_iou_v2(box1, box2):
    """
    generalized_iou
    计算广义交并比（Generalized IoU）
    返回值范围：[-1, 1]
    """
    # 计算交集
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算各区域面积
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    
    # 最小闭合区域面积
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # GIoU计算
    iou = inter_area / (union_area + 1e-6)
    giou = iou - (c_area - union_area) / (c_area + 1e-6)

    normed_giou = (1 + giou) / 2
    return normed_giou

def Hungarian_Matching(preds, gts, iou_threshold=0.5): # 这个匹配，是不是还必须考虑同类型的才能去做匹配？
    """
    定位奖励：同类别内匈牙利匹配，返回平均IoU和匹配对信息
    Args:
        preds: 预测框列表 [{'bbox_2d': [...], 'label': ...}, ...]
        gts: 真实框列表 [{'bbox_2d': [...], 'label': ...}, ...]
    """
    # 按类别分组
    categories = set(gt['label'] for gt in gts).union(set(p['label'] for p in preds))
    total_iou = 0.0
    matched_pairs = []
    
    for cat in categories:
        # 提取同类别的预测和真实框
        cat_preds = [p for p in preds if p['label'] == cat]
        cat_gts = [g for g in gts if g['label'] == cat]
        
        if not cat_preds or not cat_gts:
            continue
        
        # 构建IoU代价矩阵（1 - IoU）
        cost_matrix = 1 - np.array([[calculate_iou_v2(p['bbox_2d'], g['bbox_2d']) 
                                   for g in cat_gts] for p in cat_preds])
        
        # 匈牙利匹配
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # 记录有效匹配对
        for p_idx, g_idx in zip(pred_indices, gt_indices):
            iou = 1 - cost_matrix[p_idx][g_idx]
            if iou >= iou_threshold:
                total_iou += iou
                matched_pairs.append({
                    'pred_idx': p_idx,
                    'gt_idx': g_idx,
                    'iou': iou,
                    'category': cat
                })
    return total_iou, matched_pairs

def compute_single_image_ap(preds, gts, iou_threshold=0.5):
    """计算单张图像的近似mAP奖励"""
    # 按类别分组
    categories = set([obj['label'] for obj in preds + gts])
    aps = []
    
    for cat in categories:
        # 获取同类别的预测和GT
        cat_preds = [p for p in preds if p['label'] == cat]
        cat_gts = [g for g in gts if g['label'] == cat]
        
        # 无GT时处理逻辑
        if not cat_gts:
            if not cat_preds:
                continue  # 双方均无目标，跳过
            else:
                aps.append(0.0)  # 存在误报，AP=0
                continue
        
        # 匈牙利匹配（仅限同类别）
        iou_matrix = np.array([[calculate_iou_v2(p['bbox_2d'], g['bbox_2d']) 
                              for g in cat_gts] for p in cat_preds])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # 统计TP/FP/FN
        tp = 0
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r,c] >= iou_threshold:
                tp += 1
        fp = len(cat_preds) - tp
        fn = len(cat_gts) - tp
        
        # 计算Precision和Recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        # 简化AP计算（单阈值近似）
        ap = precision * recall
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def class_reward(pred_boxes, gt_boxes, matched_pairs, count_penalty = False, class_mapping=None):
    """类别Reward"""
    if not matched_pairs:
        return 0.0
    
    correct = 0
    for pair in matched_pairs:
        # 由于匹配时已限制同类别，此处奖励全为1
        correct += 1
    
    return correct / len(matched_pairs)

def count_penalty_reward(pred_boxes, gt_boxes, alpha=2.5, beta=0.4, eps=1e-6):
    """
    改进后的数量差异惩罚函数
    Args:
        pred_boxes: 预测框列表
        gt_boxes: 真实框列表
        alpha: 少预测惩罚系数（pred < gt时生效）
        beta: 多预测惩罚系数（pred > gt时生效）
        eps: 数值稳定性系数
    Returns:
        reward: [0,1]区间的奖励值（1为最优）
    """
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)
    
    # 处理无真实目标的情况
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    if pred_count == 0:
        return 0.0
    
    # 计算差异并区分惩罚方向
    diff = pred_count - gt_count

    alpha = alpha * (1 - 1/(1 + np.exp(-gt_count/10)))
    beta = beta * (1 + 1/(1 + np.exp(-gt_count/10)))
    
    if diff < 0:  # 预测不足（更严重）
        penalty = alpha * np.log1p(-diff) / np.log1p(gt_count + eps)
    else:         # 预测过量
        penalty = beta * np.log1p(diff) / np.log1p(gt_count + eps)
    
    # 确保奖励值在合理范围
    reward = max(0.0, 1.0 - penalty)
    return reward

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['bbox_2d'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

def accuracy_reward_count(content, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""

    reward = 0.0
    original_reward = None
    student_answer_bbox = []
    ground_truth_bbox = []
    matched_pairs = []
    # If symbolic verification failed, try string matching

    try:
        # Extract answer from solution if it has think/answer tags
        ground_truth = solution.strip()
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        student_answer = '<answer>'+student_answer+'</answer>'

        # fix format error
        student_answer = student_answer.replace("[[",'[')  
        student_answer = student_answer.replace("]]",']')  
        student_answer = student_answer.replace("\n",'')  
        # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
        ground_truth_bbox = extract_bbox(ground_truth)
        student_answer_bbox = extract_pred_bbox(student_answer)
        if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
            reward = 0.0
        else:
            student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
            original_reward = count_penalty_reward(student_answer_bbox, ground_truth_bbox)
            if original_reward > 1.0:
                reward = 1.0
            elif original_reward < 0:
                reward = 0.0
            else:
                reward = original_reward
    except Exception:
        pass  # Keep reward as 0.0 if both methods fail

    return reward

def accuracy_reward_iou(content, solution):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""

    reward = 0.0
    original_reward = None
    student_answer_bbox = []
    ground_truth_bbox = []
    matched_pairs = []
    # If symbolic verification failed, try string matching
    try:
        # Extract answer from solution if it has think/answer tags
        ground_truth = solution.strip()
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        student_answer = '<answer>'+student_answer+'</answer>'

        # fix format error
        student_answer = student_answer.replace("[[",'[')  
        student_answer = student_answer.replace("]]",']')  
        student_answer = student_answer.replace("\n",'')  
        # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
        ground_truth_bbox = extract_bbox(ground_truth)
        student_answer_bbox = extract_pred_bbox(student_answer)
        if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
            reward = 0.0
        else:
            student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates

            # total_iou, matched_pairs = Hungarian_Matching(student_answer_bbox, ground_truth_bbox)
            # original_reward = total_iou / len(matched_pairs) if len(matched_pairs) else 0.0
            # unmatched_pairs = abs(len(ground_truth_bbox) - len(matched_pairs))
            # unmatched_ratio = unmatched_pairs / len(ground_truth_bbox) if len(ground_truth_bbox) else 1e-6 # 有可能pred数量和gt数量一致，但是并不match，所以还需要额外奖励
            # original_reward = original_reward - 0.5 * unmatched_ratio

            original_reward = compute_single_image_ap(student_answer_bbox, ground_truth_bbox)
            if original_reward > 1.0:
                reward = 1.0
            elif original_reward < 0:
                reward = 0.0
            else:
                reward = original_reward
    except Exception:
        pass  # Keep reward as 0.0 if both methods fail

    return reward

def format_reward(completions):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, completions, re.DOTALL)
    return 1.0 if match else 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.45 * accuracy_reward_iou(predict_str, ground_truth) + 0.45 * accuracy_reward_count(predict_str, ground_truth) + 0.1 * format_reward(predict_str)
