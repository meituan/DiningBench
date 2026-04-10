import json
from tqdm import tqdm
import os
import concurrent.futures
import threading
import time
import re
import jieba
from datetime import datetime

def resolve_dataset_image_path(path, dataset_dir):
    if not path:
        return path
    if os.path.isfile(path):
        return os.path.normpath(path)
    if dataset_dir:
        cand = os.path.normpath(os.path.join(dataset_dir, path))
        if os.path.isfile(cand):
            return cand
    return path

def load_test_dataset(jsonl_path):
    """Load test dataset, each line is a json. Simplified dataset structure contains id, prompt, ground_truth, images, etc."""
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                dataset.append(sample)
    return dataset

def clean_pred_text(s):
    # Only take the first line of output, remove extra whitespace
    if not isinstance(s, str):
        try:
            s = s.decode('utf-8')
        except Exception:
            return ""
    return s.strip().split('\n')[0].strip().strip('"\n\r\t ,')

def clean_thinking_by_model(full_pred, model_name):
    """
    Clean thinking process based on model name, return cleaned prediction text.
    
    Args:
        full_pred: str, full model output (including thinking process)
        model_name: str, model name
    
    Returns:
        str: cleaned prediction text
    """
    if not isinstance(full_pred, str):
        try:
            full_pred = str(full_pred)
        except Exception:
            return ""
    
    import re
    
    if model_name in ["keye-vl-1.5-8b"]:
        # Extract content from <answer></answer>
        full_pred = re.sub(r'<analysis>.*?</analysis>', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        match = re.search(r'<answer>(.*?)</answer>', full_pred, flags=re.DOTALL | re.IGNORECASE)
        if match:
            full_pred = match.group(1).strip()
        else:
            # If <answer> tag is not found, keep as is
            full_pred = full_pred.strip()
        
    elif model_name in ["mimo-vl-7b-rl-2508", "minicpm-v-4.5"]:
        # Remove thinking content in <think>...</think>\n\n
        full_pred = re.sub(r'<think>.*?</think>\n\n', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        # Also handle cases without \n\n ending
        full_pred = re.sub(r'<think>.*?</think>', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        full_pred = re.sub(r'<think>.*?</think>', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        full_pred = full_pred.strip()
    elif model_name in ['qwen-3-vl-30b-a3b-thinking', 'qwen-3-vl-4b-thinking', 'qwen-3-vl-8b-thinking']:
        full_pred = re.sub(r'.*?</think>\n\n', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        full_pred = re.sub(r'.*?</think>', '', full_pred, flags=re.DOTALL | re.IGNORECASE)
        full_pred = full_pred.strip()



    boxed_match = re.search(r'\\boxed\{([^}]*)\}', full_pred)
    if boxed_match:
        full_pred = boxed_match.group(1).strip()
        full_pred = full_pred.strip()
    
    return full_pred

def evaluate_predictions(test_dataset, model_pred_jsonl):
    """
    Evaluate model prediction results using Chinese ROUGE metrics to assess final answers.
    Args:
        test_dataset: List[dict], test dataset samples
        model_pred_jsonl: str, path to model prediction results jsonl file
    Returns:
        None (directly prints evaluation results)
    """
    try:
        from rouge_chinese import Rouge
    except ImportError:
        print("Error: rouge_chinese package not found. Please install it with: pip install rouge-chinese")
        return
    
    # Initialize Chinese ROUGE scorer
    rouge = Rouge()
    
    # Read model prediction file, sort by idx
    model_preds_dict = {}
    invalid_count = 0
    with open(model_pred_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                pred_obj = json.loads(line)
                idx = pred_obj.get("idx", len(model_preds_dict))
                model_preds_dict[idx] = pred_obj
            except Exception:
                invalid_count += 1

    # Check length alignment
    n_gt = len(test_dataset)
    n_pred = len(model_preds_dict)
    paired = min(n_gt, n_pred)

    # Statistics
    answer_extracted = 0  # Number of successfully extracted answers
    gt_missing = 0  # Number of missing ground truth answers
    gt_valid_count = 0  # Number of samples with valid ground truth answers
    
    # Accumulate ROUGE scores
    rouge1_scores = {'precision': [], 'recall': [], 'fmeasure': []}
    rouge2_scores = {'precision': [], 'recall': [], 'fmeasure': []}
    rougeL_scores = {'precision': [], 'recall': [], 'fmeasure': []}
    
    for i in range(paired):
        gt_item = test_dataset[i]
        pred_item = model_preds_dict.get(i)

        if pred_item is None:
            continue

        gt_answer = None

        try:
            # Simplified dataset directly gets answer from ground_truth field
            gt_answer = gt_item.get("ground_truth", "")
        except Exception:
            gt_missing += 1
            continue

        # Get model prediction: read from model_pred field
        pred_text = pred_item.get("model_pred", "").strip()
        if not isinstance(pred_text, str):
            pred_text = str(pred_text).strip()
        
        # Parse model prediction answer
        pred_answer = pred_text.strip()
        
        # Count extraction success rate
        if pred_answer is not None and pred_answer != "":
            answer_extracted += 1
        
        # Use Chinese ROUGE to evaluate answer (requires ground truth to exist)
        if gt_answer:
            gt_valid_count += 1
            # If prediction answer is empty, use empty string
            pred_answer_text = pred_answer if pred_answer else ""
            gt_answer_text = gt_answer if gt_answer else ""
            
            # Calculate Chinese ROUGE scores
            try:
                # rouge.get_scores returns a list, take the first element
                scores_list = rouge.get_scores(' '.join(jieba.cut(pred_answer_text)), ' '.join(jieba.cut(gt_answer_text)))
                if scores_list and len(scores_list) > 0:
                    scores = scores_list[0]
                    
                    # Accumulate ROUGE-1 scores
                    rouge1_scores['precision'].append(scores['rouge-1']['p'])
                    rouge1_scores['recall'].append(scores['rouge-1']['r'])
                    rouge1_scores['fmeasure'].append(scores['rouge-1']['f'])
                    
                    # Accumulate ROUGE-2 scores
                    rouge2_scores['precision'].append(scores['rouge-2']['p'])
                    rouge2_scores['recall'].append(scores['rouge-2']['r'])
                    rouge2_scores['fmeasure'].append(scores['rouge-2']['f'])
                    
                    # Accumulate ROUGE-L scores
                    rougeL_scores['precision'].append(scores['rouge-l']['p'])
                    rougeL_scores['recall'].append(scores['rouge-l']['r'])
                    rougeL_scores['fmeasure'].append(scores['rouge-l']['f'])
            except Exception as e:
                # If calculation fails, skip this sample
                print(f"Warning: ROUGE calculation failed for sample {i}: {e}")
                continue

    # Calculate average ROUGE scores
    def calc_avg(scores_list):
        return sum(scores_list) / len(scores_list) if scores_list else 0.0
    
    rouge1_precision = calc_avg(rouge1_scores['precision'])
    rouge1_recall = calc_avg(rouge1_scores['recall'])
    rouge1_f1 = calc_avg(rouge1_scores['fmeasure'])
    
    rouge2_precision = calc_avg(rouge2_scores['precision'])
    rouge2_recall = calc_avg(rouge2_scores['recall'])
    rouge2_f1 = calc_avg(rouge2_scores['fmeasure'])
    
    rougeL_precision = calc_avg(rougeL_scores['precision'])
    rougeL_recall = calc_avg(rougeL_scores['recall'])
    rougeL_f1 = calc_avg(rougeL_scores['fmeasure'])
    
    # Other statistics
    answer_extract_rate = answer_extracted / paired if paired > 0 else 0.0
    fail_load_count = invalid_count + max(n_gt - n_pred, 0)
    invalid_ratio = fail_load_count / n_gt if n_gt > 0 else 0.0
    gt_missing_ratio = gt_missing / paired if paired > 0 else 0.0

    # Print evaluation results
    print(f"rouge1_precision: {rouge1_precision:.4f}")
    print(f"rouge1_recall: {rouge1_recall:.4f}")
    print(f"rouge1_f1: {rouge1_f1:.4f}")
    print(f"rouge2_precision: {rouge2_precision:.4f}")
    print(f"rouge2_recall: {rouge2_recall:.4f}")
    print(f"rouge2_f1: {rouge2_f1:.4f}")
    print(f"rougeL_precision: {rougeL_precision:.4f}")
    print(f"rougeL_recall: {rougeL_recall:.4f}")
    print(f"rougeL_f1: {rougeL_f1:.4f}")
    print(f"answer_extract_rate: {answer_extract_rate:.4f}")
    print(f"invalid_ratio: {invalid_ratio:.4f}")
    print(f"gt_missing_ratio: {gt_missing_ratio:.4f}")
    print(f"answer_extracted: {answer_extracted}")
    print(f"total_eval: {paired}")
    print(f"fail_load_count: {fail_load_count}")
    print(f"gt_missing_count: {gt_missing}")
    print(f"gt_valid_count: {gt_valid_count}")
    print(f"test_count: {n_gt}")
    print(f"pred_count: {n_pred}")

def evaluate_api_call(pred_answer, gt_answer, question, api_url, api_key, model_name, key_manager=None, max_retries=3):
    """
    Call evaluation model API to judge whether predicted answer and ground truth are consistent.
    
    Args:
        pred_answer: str, model predicted answer
        gt_answer: str, ground truth answer
        question: str, original question
        api_url: str, evaluation model API URL
        api_key: str or list, API KEY
        model_name: str, evaluation model name
        key_manager: APIKeyManager, API key manager
        max_retries: int, maximum retry count
    
    Returns:
        str: evaluation result, should be "是" or "否" or "一致" or "不一致" etc.
    """
    from openai import OpenAI
    
    prompt_en = f"""You are an intelligent evaluation assistant. Please judge whether the predicted answer is correct based on the [Question] and [Ground Truth].

### Evaluation Rules
1. **Semantic Equivalence**: As long as the core meaning expressed by the predicted answer is consistent with the ground truth, it is considered correct.
2. **Yes/No Question Compatibility**: For yes/no questions, if the ground truth is "Yes" or "No", and the predicted answer is a complete statement (e.g., confirming a fact in the question), as long as the logic is consistent, it must be judged as "Yes".
   - *Example*: Question "Is it fried?", Ground truth "Yes", Predicted answer "It has been fried" -> Judge as "Yes".
   - *Example*: Question "Is it toxic?", Ground truth "No", Predicted answer "Non-toxic" -> Judge as "Yes".
3. **Ignore Redundancy**: Ignore redundant modal particles, punctuation, or irrelevant explanatory text in the predicted answer.

### Data to Evaluate
[Question]:
{question}

[Ground Truth]:
{gt_answer}

[Predicted Answer]:
{pred_answer}

### Output
Please answer with only one word: "Yes" or "No"."""


    prompt = f"""你是一个智能评估助手。请根据【问题】和【标准答案】，判断【预测答案】是否正确。

### 判定规则
1. **语义等价**：只要预测答案表达的核心含义与标准答案一致，即为正确。
2. **是非题兼容**：对于是非类问题，如果标准答案是“是”或“否”，而预测答案是完整的陈述句（例如确认了问题中的事实），只要逻辑一致，必须判定为“是”。
   - *示例*：问题“是否油炸？”，标准答案“是”，预测答案“经历了油炸” -> 判定为“是”。
   - *示例*：问题“是否有毒？”，标准答案“没有”，预测答案“无毒” -> 判定为“是”。
3. **忽略冗余**：忽略预测答案中多余的语气词、标点或无关的解释性文字。

### 待评估数据
【问题】：
{question}

【标准答案】：
{gt_answer}

【预测答案】：
{pred_answer}

### 输出
请只回答一个字：“是” 或 “否”。"""
    
    for i in range(max_retries):
        used_api_key = None
        try:
            if key_manager is not None:
                used_api_key = key_manager.get_api_key()
            elif isinstance(api_key, list):
                idx = int(time.time() * 1000) % len(api_key)
                used_api_key = api_key[idx]
            else:
                used_api_key = api_key

            client = OpenAI(base_url=api_url, api_key=used_api_key)
            messages = [{
                "role": "user",
                "content": prompt
            }]
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            result = resp.choices[0].message.content.strip()
            return result
        except Exception as ex:
            last_exception = ex
            if i == max_retries - 1:
                print(f"Evaluation API call exception (retry {i + 1}/{max_retries}): {ex}")
                return "__EVAL_ERROR__"
            else:
                time.sleep(1)
                continue

def parse_eval_result(eval_text):
    """
    Parse evaluation result text to determine if it indicates a match.
    
    Args:
        eval_text: str, the output text from the evaluation model
    
    Returns:
        bool: True indicates match/consistent, False indicates mismatch
    """
    if not isinstance(eval_text, str):
        return False
    
    eval_text = eval_text.strip().lower()
    
    # Check for keywords indicating "match" or "consistent"
    positive_keywords = ["yes", "true", "Yes", "correct", "match", "same", "是", "一致", "相同", "等价", "正确"]
    negative_keywords = ["no", "false", "No", "incorrect", "different", "not match", "否", "不一致", "不同", "错误"]
    
    # Check negative keywords first
    for keyword in negative_keywords:
        if keyword in eval_text:
            return False
    
    # Check positive keywords
    for keyword in positive_keywords:
        if keyword in eval_text:
            return True
    
    # If no match, default to False
    return False

def evaluate_predictions_with_model(test_dataset, model_pred_jsonl, evaluate_api_url, evaluate_api_key, evaluate_model_name, evaluate_max_workers=16, evaluate_rpm=None, output_eval_jsonl=None):
    """
    Use large model to evaluate consistency between prediction results and ground truth.
    
    Args:
        test_dataset: List[dict], test dataset samples
        model_pred_jsonl: str, path to model prediction results jsonl file
        evaluate_api_url: str, evaluation model API URL
        evaluate_api_key: str or list, evaluation model API KEY
        evaluate_model_name: str, evaluation model name
        evaluate_max_workers: int, concurrent evaluation thread count
        evaluate_rpm: str or None, evaluation model RPM limit
        output_eval_jsonl: str or None, path to save evaluation results, if None then save to model_pred_jsonl
    
    Returns:
        None (directly prints evaluation results)
    """
    # Parse evaluation API keys and setup key_manager
    eval_api_keys, eval_rpms, eval_key_manager = parse_api_keys_and_setup_manager(
        evaluate_api_key,
        rpm=evaluate_rpm,
        infer=True,
        default_rpm=20
    )
    
    # Determine output file path
    output_file = output_eval_jsonl
    
    # Load existing evaluation results from evaluation output file
    already_evaluated = {}  # idx -> (eval_match, eval_result)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        eval_obj = json.loads(line)
                        idx = eval_obj.get("idx")
                        eval_match = eval_obj.get("eval_match")
                        eval_result = eval_obj.get("eval_result", "")
                        # If evaluation result exists and is not an error result, record it
                        if idx is not None and eval_match is not None and isinstance(eval_result, str) and "__EVAL_ERROR__" not in eval_result:
                            already_evaluated[idx] = (eval_match, eval_result)
                    except Exception:
                        continue
        except Exception as ex:
            print(f"Error reading existing evaluation results file: {ex}")
    
    print(f"Loaded {len(already_evaluated)} existing evaluation results")
    
    # Read model prediction file, sort by idx
    model_preds_dict = {}
    invalid_count = 0
    with open(model_pred_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                pred_obj = json.loads(line)
                idx = pred_obj.get("idx", len(model_preds_dict))
                model_preds_dict[idx] = pred_obj
            except Exception:
                invalid_count += 1

    # Check length alignment
    n_gt = len(test_dataset)
    n_pred = len(model_preds_dict)
    paired = min(n_gt, n_pred)

    # Prepare evaluation tasks (only evaluate unevaluated or failed samples)
    eval_tasks = []
    for i in range(paired):
        # If valid evaluation result already exists, skip
        if i in already_evaluated:
            continue
            
        gt_item = test_dataset[i]
        pred_item = model_preds_dict.get(i)

        if pred_item is None:
            continue

        gt_answer = None
        question = None
        try:
            # Simplified dataset directly gets from fields
            gt_answer = gt_item.get("ground_truth", "")
            # Extract question from prompt (remove fixed ending)
            prompt = gt_item.get("prompt", "")
            fixed_ending = "\n仅输出最终答案，不要包含任何解释、分析、前言或总结。"
            fixed_ending_en = "\nOutput only the final answer, without any explanation, analysis, preface, or summary."
            if prompt.endswith(fixed_ending_en):
                question = prompt[:-len(fixed_ending_en)].strip()
            elif prompt.endswith(fixed_ending):
                question = prompt[:-len(fixed_ending)].strip()
            else:
                question = prompt.strip()
        except Exception:
            continue

        if not gt_answer:
            continue

        # Get model prediction
        pred_text = pred_item.get("model_pred", "").strip()
        if not isinstance(pred_text, str):
            pred_text = str(pred_text).strip()
        
        pred_answer = pred_text.strip()
        
        # Error predictions are also included in evaluation, but marked for special handling
        is_error = "__INFER_ERROR__" in pred_answer or not pred_answer
        
        eval_tasks.append((i, pred_answer, gt_answer, question, is_error))
    
    print(f"Preparing to evaluate {len(eval_tasks)} samples (skipping {len(already_evaluated)} existing evaluation results)")
    
    # Concurrent evaluation
    def single_eval(args):
        idx, pred_answer, gt_answer, question, is_error = args
        # If it's an error prediction, directly mark as inconsistent, no need to call API
        if is_error:
            return idx, False, "__INFER_ERROR__"
        
        eval_result = evaluate_api_call(
            pred_answer,
            gt_answer,
            question,
            evaluate_api_url,
            eval_api_keys[0] if (len(eval_api_keys) == 1 and eval_key_manager is None) else eval_api_keys,
            evaluate_model_name,
            key_manager=eval_key_manager
        )
        is_match = parse_eval_result(eval_result)
        return idx, is_match, eval_result
    
    # If there are samples to evaluate, perform concurrent evaluation
    eval_results = []
    if eval_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=evaluate_max_workers) as exe:
            futs = [exe.submit(single_eval, task) for task in eval_tasks]
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Evaluating"):
                eval_results.append(f.result())
    else:
        print("All samples have been evaluated, no need to re-evaluate")
    
    # Add new evaluation results to dictionary
    eval_results_dict = {}
    for idx, is_match, eval_result in eval_results:
        eval_results_dict[idx] = {
            "eval_match": is_match,
            "eval_result": eval_result
        }
    
    # Merge existing evaluation results
    for idx, (eval_match, eval_result) in already_evaluated.items():
        eval_results_dict[idx] = {
            "eval_match": eval_match,
            "eval_result": eval_result
        }
    
    # Update model_preds_dict, add evaluation fields
    for idx, pred_obj in model_preds_dict.items():
        if idx in eval_results_dict:
            pred_obj["eval_match"] = eval_results_dict[idx]["eval_match"]
            pred_obj["eval_result"] = eval_results_dict[idx]["eval_result"]
    
    # Save updated results to file
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        # Write after sorting by idx
        sorted_indices = sorted(model_preds_dict.keys())
        for idx in sorted_indices:
            fout.write(json.dumps(model_preds_dict[idx], ensure_ascii=False) + "\n")
    print(f"Evaluation results saved to: {output_file}")
    
    # Statistics (including existing and new evaluation results)
    all_eval_results = []
    # Add new evaluation results
    for idx, is_match, eval_result in eval_results:
        all_eval_results.append((idx, is_match, eval_result))
    # Add existing evaluation results
    for idx, (eval_match, eval_result) in already_evaluated.items():
        all_eval_results.append((idx, eval_match, eval_result))
    
    total_eval = len(all_eval_results)
    correct_count = sum(1 for _, is_match, _ in all_eval_results if is_match)
    error_count = sum(1 for _, _, eval_result in all_eval_results if isinstance(eval_result, str) and "__EVAL_ERROR__" in eval_result)
    infer_error_count = sum(1 for _, _, eval_result in all_eval_results if isinstance(eval_result, str) and "__INFER_ERROR__" in eval_result)
    new_eval_count = len(eval_results)
    already_eval_count = len(already_evaluated)
    
    accuracy = correct_count / total_eval if total_eval > 0 else 0.0
    error_rate = error_count / total_eval if total_eval > 0 else 0.0
    
    # Print evaluation results
    print(f"model_evaluate_accuracy: {accuracy:.4f}")
    print(f"model_evaluate_correct: {correct_count}")
    print(f"model_evaluate_total: {total_eval}")
    print(f"model_evaluate_error_count: {error_count}")
    print(f"model_evaluate_error_rate: {error_rate:.4f}")
    print(f"infer_error_count: {infer_error_count}")
    print(f"new_eval_count: {new_eval_count}")
    print(f"already_eval_count: {already_eval_count}")
    print(f"test_count: {n_gt}")
    print(f"pred_count: {n_pred}")
    print(f"invalid_count: {invalid_count}")


class APIKeyManager:
    def __init__(self, api_keys, rpms):
        """
        api_keys: List[str]
        rpms: List[int], corresponds to api_keys in order, maximum rpm for each key
        """
        self.locks = [threading.Lock() for _ in api_keys]
        self.last_times = [0] * len(api_keys)
        self.api_keys = api_keys
        self.rpms = rpms
        self.intervals = [60 / rpm if rpm > 0 else 1e6 for rpm in rpms]
        self.next_available_time = [0] * len(api_keys)
        self.counter = 0
        self.n = len(api_keys)
        self.timer_lock = threading.Lock()

    def get_api_key(self):
        """
        Get next available API Key and perform rpm-level flow control.
        """
        while True:
            with self.timer_lock:
                idx = self.counter
                self.counter = (self.counter + 1) % self.n
            now = time.time()
            # Try to check each key in turn
            for att in range(self.n):
                key_idx = (idx + att) % self.n
                with self.locks[key_idx]:
                    wait = self.next_available_time[key_idx] - now
                    if wait > 0:
                        continue
                    t_gap = now - self.last_times[key_idx]
                    min_gap = self.intervals[key_idx]
                    if t_gap < min_gap:
                        # Meets rate limit, need to wait
                        wait_time = min_gap - t_gap
                        self.next_available_time[key_idx] = now + wait_time
                        continue
                    # Allow call
                    self.last_times[key_idx] = now
                    self.next_available_time[key_idx] = now + min_gap
                    return self.api_keys[key_idx]
            # If all keys need to wait, wait for the shortest next_available_time
            soonest = min(self.next_available_time)
            sleep_time = soonest - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0.05)

def vllm_api_infer_func(model_input, image_paths, max_retries=3, api_url="http://127.0.0.1:8000/generate", api_key="EMPTY", model_name="judge", key_manager=None):
    """
    Inference interface: input prompt and images, output food_name text.
    Supports multiple image inputs, image_paths should be a list (length controlled by num_images_idxs parameter), default is single image can directly pass str.
    Supports API KEY as list (key_manager parameter takes precedence).
    """
    import base64
    from openai import OpenAI
    
    if not image_paths:
        raise ValueError(f"Image path is empty: {image_paths}")
    
    # Support single or multiple images
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    img_base64_list = []
    for path in image_paths:
        if path is None:
            raise ValueError(f"Image path is None: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image does not exist: {path}")
        try:
            with open(path, "rb") as fin:
                img_base64 = base64.b64encode(fin.read()).decode("utf-8")
                img_base64_list.append(img_base64)
        except Exception as ex:
            raise IOError(f"Failed to read image file: {path}, error: {ex}") from ex
    
    prompt = model_input
    
    last_exception = None
    for i in range(max_retries):
        used_api_key = None
        try:
            if key_manager is not None:
                used_api_key = key_manager.get_api_key()
            elif isinstance(api_key, list):
                idx = int(time.time() * 1000) % len(api_key)
                used_api_key = api_key[idx]
            else:
                used_api_key = api_key

            client = OpenAI(base_url=api_url, api_key=used_api_key)
            # Add multiple images sequentially
            image_contents = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} 
                for b64 in img_base64_list
            ]
            # Ensure image content comes first, then add text
            messages = [{
                "role": "user",
                "content": image_contents + [
                    {"type": "text", "text": prompt}
                ]
            }]

            max_tokens = 16384 - 2048  # Other models: 14336
            
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as ex:
            last_exception = ex
            if i == max_retries - 1:
                # Last retry failed, raise exception
                print(f"Inference exception (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}")
                raise RuntimeError(f"Inference exception (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}") from ex
            else:
                print(f"Inference exception (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}, retrying...")
                time.sleep(1)
                continue

def load_existing_results(output_jsonl):
    """
    Load existing output file, return set of processed indices and list of existing results.
    
    Args:
        output_jsonl: str, output file path
    
    Returns:
        tuple: (processed_indices, existing_results)
            - processed_indices: set[int], set of processed indices
            - existing_results: List[dict], list of existing results
    """
    processed_indices = set()
    existing_results = []
    if os.path.exists(output_jsonl):
        try:
            with open(output_jsonl, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pred_obj = json.loads(line)
                        idx = pred_obj.get("idx")
                        model_pred = pred_obj.get("model_pred", "")
                        if idx is not None and model_pred != "" and "__INFER_ERROR__" not in model_pred:
                            processed_indices.add(idx)
                            existing_results.append(pred_obj)
                    except Exception:
                        continue
        except Exception as ex:
            print(f"Error reading existing results file: {ex}")
    return processed_indices, existing_results

def count_all_predictions(output_jsonl, total_samples):
    """
    Check if all samples in output file have prediction results (including error results).
    
    Args:
        output_jsonl: str, output file path
        total_samples: int, total number of samples
    
    Returns:
        int: number of samples with prediction results (including error results)
    """
    all_predicted_indices = set()
    if os.path.exists(output_jsonl):
        try:
            with open(output_jsonl, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pred_obj = json.loads(line)
                        idx = pred_obj.get("idx")
                        model_pred = pred_obj.get("model_pred", "")
                        if idx is not None and model_pred != "" and "__INFER_ERROR__" not in model_pred:
                            all_predicted_indices.add(idx)
                    except Exception:
                        continue
        except Exception as ex:
            print(f"Error reading existing results file: {ex}")
    return len(all_predicted_indices)

def run_inference(test_dataset, infer_func, output_jsonl, max_workers=16, num_images_idxs=None, resume=False, model_name="judge", dataset_dir=None):
    """
    Input prompt and images to make text predictions.
    Note: In simplified dataset (stage3_core_en.jsonl), the images field already contains all required image paths,
    so the num_images_idxs parameter is no longer used (kept only for interface compatibility).
    num_images_idxs: list of int, deprecated (images field in simplified dataset already specifies required images)
    resume: bool, whether to resume from existing output file, skip already processed samples
    model_name: str, model name, used for cleaning thinking process
    """
    def prepare_args(args):
        idx, sample = args

        # Simplified dataset directly gets image path list from images field
        images = sample.get("images", [])
        if not images:
            # If images field doesn't exist, try to construct from id (compatible with minimal format)
            dish_id = sample.get("id", "")
            if dish_id != "" and dish_id is not None:
                try:
                    dish_id_int = int(dish_id)
                    images = [f"images/{dish_id_int:05d}.jpg"]
                except Exception:
                    dish_id_str = str(dish_id).strip()
                    if dish_id_str.isdigit():
                        images = [f"images/{dish_id_str.zfill(5)}.jpg"]
                    else:
                        images = []
            else:
                images = []
        
        # Get prompt from prompt field (already contains fixed ending)
        prompt = sample.get("prompt", "")
        if not prompt:
            prompt = ""

        images = [resolve_dataset_image_path(p, dataset_dir) for p in images]

        # Process image paths: if only one image, pass string directly; otherwise pass list
        img_paths = images[0] if len(images) == 1 else images
        
        # Get reasoning and ground_truth from fields
        reasoning = sample.get("reasoning", "")
        final_answer = sample.get("ground_truth", "")
        label = f"reasoning: {reasoning}\nanswer: {final_answer}"
        return idx, prompt, img_paths, label, model_name

    def single_infer(args):
        idx, prompt, img_paths, label, model_name = args
        try:
            full_pred = infer_func(prompt, img_paths)
        except Exception as ex:
            full_pred = f"__INFER_ERROR__: {ex}"
        
        # Save full output
        model_full_pred = full_pred
        
        # Clean thinking process based on model name
        if "__INFER_ERROR__" not in full_pred:
            model_pred = clean_thinking_by_model(full_pred, model_name)
        else:
            model_pred = full_pred
        
        d = {
            "idx": idx, 
            "model_pred": model_pred, 
            "model_full_pred": model_full_pred,
            "model_input": prompt, 
            "label": label
        }
        # Write currently used images
        if isinstance(img_paths, list):
            d["image_paths"] = img_paths
            if len(img_paths) == 1:
                d["image_path"] = img_paths[0]
        else:
            d["image_path"] = img_paths
        return d

    # If resume is enabled, load existing results and filter processed samples
    processed_indices = set()
    existing_results = []
    if resume:
        processed_indices, existing_results = load_existing_results(output_jsonl)
        print(f"Resume mode: processed {len(processed_indices)} samples, {len(test_dataset) - len(processed_indices)} samples remaining")
    
    # Filter out processed samples
    remaining_dataset = []
    remaining_indices = []
    for i, sample in enumerate(test_dataset):
        if i not in processed_indices:
            remaining_dataset.append(sample)
            remaining_indices.append(i)
    
    if not remaining_dataset:
        print("All samples have been processed, no inference needed")
        return count_all_predictions(output_jsonl, len(test_dataset))
    
    print(f"Multi-threaded inference, thread count: {max_workers}, samples to process: {len(remaining_dataset)} (simplified dataset: image paths already obtained from images field)")
    # Use original index (i) instead of index in remaining_dataset
    args_list = [prepare_args((remaining_indices[j], x)) for j, x in enumerate(remaining_dataset)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = [exe.submit(single_infer, args) for args in args_list]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Inferring"):
            results.append(f.result())
    
    # Merge old and new results
    all_results = existing_results + results
    all_results = sorted(all_results, key=lambda x: x['idx'])
    
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for ob in all_results:
            fout.write(json.dumps(ob, ensure_ascii=False) + "\n")
    print(f"Inference results saved to: {output_jsonl}, total {len(all_results)} entries (new {len(results)} entries)")
    
    # Return number of processed samples (including error results)
    return count_all_predictions(output_jsonl, len(test_dataset))

def parse_api_keys_and_setup_manager(api_key, rpm=None, infer=False, default_rpm=20):
    """
    Parse API keys and RPM parameters, and setup APIKeyManager.
    
    Args:
        api_key: API key, can be string, list, etc., supports comma-separated or Python list string
        rpm: RPM limit, can be string, list, int or None, supports comma-separated or Python list string
        infer: whether to perform inference, used to decide whether to create key_manager
        default_rpm: default RPM value, used when rpm is None
    
    Returns:
        tuple: (api_keys, rpms, key_manager)
            - api_keys: List[str], parsed API keys list
            - rpms: List[int], parsed RPM list
            - key_manager: APIKeyManager or None
    """
    import ast
    
    # Parse api_keys
    api_keys = api_key
    # Try to parse as list
    if isinstance(api_keys, str):
        try:
            api_keys_eval = ast.literal_eval(api_keys)
            if isinstance(api_keys_eval, list):
                api_keys = [str(x) for x in api_keys_eval]
            elif isinstance(api_keys_eval, tuple):
                api_keys = [str(x) for x in api_keys_eval]
            else:
                api_keys = [api_keys]
        except Exception:
            # Support comma-separated string
            api_keys = [x.strip() for x in api_keys.split(",") if x.strip()]
    elif isinstance(api_keys, list):
        pass
    else:
        api_keys = [str(api_keys)]

    # Parse rpms
    if rpm is None:
        rpms = [default_rpm] * len(api_keys)
    else:
        rpm_arg = rpm
        # Try to parse as list
        if isinstance(rpm_arg, str):
            try:
                rpm_eval = ast.literal_eval(rpm_arg)
                if isinstance(rpm_eval, list):
                    rpms = [int(x) for x in rpm_eval]
                else:
                    rpms = [int(rpm_eval)] * len(api_keys)
            except Exception:
                # Support comma-separated string
                rpms = [int(x) for x in rpm_arg.split(",") if x.strip()]
                if len(rpms) == 1 and len(api_keys) > 1:
                    rpms = rpms * len(api_keys)
        elif isinstance(rpm_arg, list):
            rpms = [int(x) for x in rpm_arg]
        else:
            rpms = [int(rpm_arg)] * len(api_keys)
    # Align length, if rpms is shorter than api_keys, pad with last value
    if len(rpms) < len(api_keys):
        rpms = rpms + [rpms[-1]] * (len(api_keys) - len(rpms))
    elif len(rpms) > len(api_keys):
        rpms = rpms[:len(api_keys)]

    # Create key_manager
    key_manager = None
    if infer and (len(api_keys) > 1 or (rpm is not None and len(rpms) > 1)):
        key_manager = APIKeyManager(api_keys, rpms)
    
    return api_keys, rpms, key_manager

def parse_num_images_idxs(num_images_idxs):
    """
    Process image index list.
    
    Args:
        num_images_idxs: can be None, string (e.g., "[0,2,3]" or "0,1,2"), etc.
    
    Returns:
        list[int]: image index list
    """
    import ast
    if num_images_idxs is None:
        return [0]
    else:
        try:
            # Support python list string, e.g., "[0,2,3]"
            num_images_idxs = ast.literal_eval(num_images_idxs)
            if isinstance(num_images_idxs, int):
                num_images_idxs = [num_images_idxs]
            elif not isinstance(num_images_idxs, list):
                num_images_idxs = [int(num_images_idxs)]
            else:
                num_images_idxs = [int(x) for x in num_images_idxs]
        except Exception:
            # e.g., "0,1,2"
            num_images_idxs = [int(x) for x in num_images_idxs.split(",") if str(x).strip() != '']
    return num_images_idxs

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Script start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action='store_true', default=False, help="Whether to perform inference")
    parser.add_argument("--evaluate", action='store_true', default=False, help="Whether to evaluate")
    parser.add_argument("--api_url", type=str, default="http:///v1", help="Inference API URL")
    parser.add_argument("--api_key", type=str, default="test", help="API KEY/Multiple API KEYs available, comma-separated or python list/list string")
    parser.add_argument("--rpm", type=str, default=None, help="RPM for each api_key, e.g.: 20 or [20,15,30] or 20,20,20")
    parser.add_argument("--test_jsonl_path", type=str, default='', help="Test set jsonl path")
    parser.add_argument("--output_pred_jsonl", type=str, default="./evaluate_results/output_pred.jsonl", help="Inference output jsonl path")
    parser.add_argument("--sample_num", type=int, default=10000, help="Number of samples for inference or evaluation")
    parser.add_argument("--model_name", type=str, default="judge", help="Model name")
    parser.add_argument("--max_workers", type=int, default=16, help="Concurrent thread count")
    parser.add_argument("--num_images_idxs", type=str, default='0', help="List of image indices, e.g., [0,1] or 0,1, indicating image idx used for each input (starting from 0). Default uses only the first image.")
    parser.add_argument("--resume", action='store_true', default=False, help="Whether to resume from existing output file, skip already processed samples")
    parser.add_argument("--max_retry_loops", type=int, default=5, help="Maximum loop count, if inference fails to process all samples, automatically resume and infer again until maximum loop count is reached or all samples are processed")
    parser.add_argument("--model_evaluate", action='store_true', default=False, help="Whether to use large model to evaluate consistency between predictions and ground truth")
    parser.add_argument("--evaluate_api", type=str, default="http://v1", help="Evaluation model API URL")
    parser.add_argument("--evaluate_api_key", type=str, default="test", help="Evaluation model API KEY/Multiple API KEYs available, comma-separated or python list/list string")
    parser.add_argument("--evaluate_model_name", type=str, default="judge", help="Evaluation model name")
    parser.add_argument("--evaluate_max_workers", type=int, default=16, help="Concurrent thread count during evaluation")
    parser.add_argument("--evaluate_rpm", type=str, default=None, help="RPM for each api_key of evaluation model, e.g.: 20 or [20,15,30] or 20,20,20")
    parser.add_argument("--output_eval_jsonl", type=str, default=None, help="Path to save evaluation results, if not specified then save to original prediction file")
    args = parser.parse_args()
    # Process image index list
    num_images_idxs = parse_num_images_idxs(args.num_images_idxs)

    test_dataset = load_test_dataset(args.test_jsonl_path)
    test_dataset = test_dataset[:args.sample_num]

    api_keys, rpms, key_manager = parse_api_keys_and_setup_manager(
        args.api_key, 
        rpm=args.rpm, 
        infer=args.infer, 
        default_rpm=20
    )

    if args.infer:
        total_samples = len(test_dataset)
        loop_count = 0
        is_first_loop = True
        
        while loop_count < args.max_retry_loops:
            loop_count += 1
            print(f"\n{'='*60}")
            print(f"Inference loop {loop_count}/{args.max_retry_loops}")
            print(f"{'='*60}")
            
            # First loop uses user-specified resume parameter, subsequent loops force resume=True
            current_resume = args.resume if is_first_loop else True
            is_first_loop = False
            
            processed_count = run_inference(
                test_dataset,
                lambda m, imgs: vllm_api_infer_func(
                    m, imgs,
                    max_retries=1,
                    api_url=args.api_url,
                    api_key=api_keys[0] if (len(api_keys) == 1 and key_manager is None) else api_keys,
                    model_name=args.model_name,
                    key_manager=key_manager
                ),
                args.output_pred_jsonl,
                max_workers=args.max_workers,
                num_images_idxs=num_images_idxs,
                resume=current_resume,
                model_name=args.model_name,
                dataset_dir=os.path.dirname(os.path.abspath(args.test_jsonl_path)),
            )
            
            # Check if all samples have been processed
            if processed_count >= total_samples:
                print(f"\nAll samples have been processed! Processed {processed_count}/{total_samples} samples")
                break
            else:
                remaining = total_samples - processed_count
                print(f"\nCurrently processed {processed_count}/{total_samples} samples, {remaining} samples remaining")
                if loop_count < args.max_retry_loops:
                    print(f"Will continue processing remaining samples in next loop...")
                else:
                    print(f"Reached maximum loop count {args.max_retry_loops}, stopping inference")
                    print(f"Still {remaining} samples not processed")
    if args.evaluate:
        if args.model_evaluate:
            # Use large model for evaluation
            evaluate_predictions_with_model(
                test_dataset,
                args.output_pred_jsonl,
                args.evaluate_api,
                args.evaluate_api_key,
                args.evaluate_model_name,
                args.evaluate_max_workers,
                args.evaluate_rpm,
                args.output_eval_jsonl
            )
        else:
            # Use ROUGE for evaluation
            evaluate_predictions(test_dataset, args.output_pred_jsonl)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Script end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_time}")
