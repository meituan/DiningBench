import json
from tqdm import tqdm
import os
import concurrent.futures
import threading
import time
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
    """Load test dataset, each line is a json containing fields like id, prompt, ground_truth, standard_image, user_images, etc."""
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

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
            # If <answer> tag is not found, keep original
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
    Evaluate model predictions, calculate accuracy (acc) and ratio of failed predictions.
    Args:
        test_dataset: List[dict], test dataset samples, each sample should contain 'ground_truth' field (option letter)
        model_pred_jsonl: str, path to model prediction jsonl file
    Returns:
        result: dict, containing acc and invalid_ratio
    """
    # Read model prediction file, sorted by idx
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

    matched = 0
    cannot_read_option_count = 0  # Count number of cases where valid option letters (A/B/C/D, etc.) cannot be read
    for i in range(paired):
        gt_item = test_dataset[i]
        pred_item = model_preds_dict.get(i)

        if pred_item is None:
            continue

        # Prioritize reading ground truth from output file (if exists)
        gt_answer = pred_item.get("gt_answer", "")
        
        # If ground truth is not saved in output file, get it from test dataset
        if not gt_answer:
            gt_answer = gt_item.get("ground_truth", "").strip().upper()

        # Get model prediction: read from model_pred field, consistent with run_inference output
        pred_answer = pred_item.get("model_pred", "").strip()
        if not isinstance(pred_answer, str):
            pred_answer = str(pred_answer).strip()

        if not pred_answer or "__INFER_ERROR__" in pred_answer:
            cannot_read_option_count += 1
            continue
        
        # Process prediction result using clean_pred_text function
        pred_text_cleaned = pred_answer
        pred_answer = pred_text_cleaned
        
        # Only take first character (option letter), convert to uppercase
        if pred_answer:
            pred_answer = pred_answer.upper().strip()
            # Extract first alphabetic character
            for char in pred_answer:
                if char.isalpha():
                    pred_answer = char
                    break
            else:
                pred_answer = ""
        else:
            pred_answer = ""
        
        # Check if valid option letter (A-Z) can be read
        if not pred_answer or pred_answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            cannot_read_option_count += 1

        if pred_answer == gt_answer and pred_answer != '':
            matched += 1

        

    acc = matched / paired if paired > 0 else 0.0
    # Failed loading includes json deserialization failures and cases where prediction file length is less than test set length
    fail_load_count = invalid_count + max(n_gt - n_pred, 0)
    invalid_ratio = fail_load_count / n_gt if n_gt > 0 else 0.0
    # Calculate probability of not being able to read valid option letters (A/B/C/D, etc.)
    cannot_read_option_ratio = cannot_read_option_count / paired if paired > 0 else 0.0

    print(f"acc: {acc:.4f}")
    print(f"cannot_read_option_ratio: {cannot_read_option_ratio:.4f}")
    print(f"cannot_read_option_count: {cannot_read_option_count}")
    print(f"invalid_ratio: {invalid_ratio:.4f}")
    print(f"matched: {matched}")
    print(f"total_eval: {paired}")
    print(f"fail_load_count: {fail_load_count}")
    print(f"test_count: {n_gt}")
    print(f"pred_count: {n_pred}")


class APIKeyManager:
    def __init__(self, api_keys, rpms):
        """
        api_keys: List[str]
        rpms: List[int], corresponding to api_keys in order, maximum rpm for each key
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
        Get next available API Key and perform rpm-level rate limiting.
        """
        while True:
            with self.timer_lock:
                idx = self.counter
                self.counter = (self.counter + 1) % self.n
            now = time.time()
            # Try checking each key in sequence
            for att in range(self.n):
                key_idx = (idx + att) % self.n
                with self.locks[key_idx]:
                    wait = self.next_available_time[key_idx] - now
                    if wait > 0:
                        continue
                    t_gap = now - self.last_times[key_idx]
                    min_gap = self.intervals[key_idx]
                    if t_gap < min_gap:
                        # Rate limit met, need to wait
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
    Supports multiple image inputs, image_paths should be a list (length controlled by num_images_idxs parameter), default is str when only one image is used.
    Supports API KEY as list (key_manager parameter takes priority).
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
            
            max_tokens = 16384 - 2048

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
                print(f"Inference error (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}")
                raise RuntimeError(f"Inference error (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}") from ex
            else:
                print(f"Inference error (retry {i + 1}/{max_retries}): {ex}, image_paths: {image_paths}, retrying...")
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
    Input prompt and images at specified indices to predict food_name text (first line only).
    Dataset should contain fields: prompt, ground_truth, standard_image, user_images
    num_images_idxs: list of int, specify which images to use, 0 means standard image, 1+ means user images (in user_images order), e.g. [0,1] means use standard image and first user image, if None then only use standard image (equivalent to [0]).
    resume: bool, whether to resume from existing output file, skip already processed samples
    """
    def prepare_args(args):
        idx, sample = args

        # Directly use prompt from dataset (already in English)
        prompt = sample.get("prompt", "")
        
        # Directly use ground_truth from dataset (option letter)
        gt_letter = sample.get("ground_truth", "").strip().upper()
        
        # Build image path list: standard image + user images
        images_path = []
        standard_image = sample.get("standard_image", "")
        if standard_image:
            images_path.append(standard_image)
        
        user_images = sample.get("user_images", [])
        if isinstance(user_images, list):
            images_path.extend(user_images)
        elif user_images:
            images_path.append(user_images)

        # Select images based on num_images_idxs
        # num_images_idxs: [0] means only standard image, [0, 1] means standard image and first user image
        indices = []
        for images_idx in num_images_idxs:
            if images_idx == 0:
                # Standard image index is 0
                if len(images_path) > 0:
                    indices.append(0)
            else:
                # User image indices start from 1 (because 0 is standard image)
                user_img_idx = images_idx
                if user_img_idx < len(images_path):
                    indices.append(user_img_idx)
        
        if len(indices) == 0:
            # If not specified or invalid, default to standard image
            if len(images_path) > 0:
                indices = [0]

        img_paths = [images_path[i] for i in indices if i < len(images_path)]
        if len(img_paths) == 0:
            # If still no images, at least use standard image
            if len(images_path) > 0:
                img_paths = [images_path[0]]
        
        # If only one image, return string directly
        if len(img_paths) == 1:
            img_paths = img_paths[0]

        if isinstance(img_paths, list):
            img_paths = [resolve_dataset_image_path(p, dataset_dir) for p in img_paths]
        else:
            img_paths = resolve_dataset_image_path(img_paths, dataset_dir)
        
        return idx, prompt, img_paths, gt_letter, model_name

    def single_infer(args):
        idx, prompt, img_paths, gt_letter, model_name = args
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
            "gt_answer": gt_letter, 
            "model_pred": model_pred, 
            "model_full_pred": model_full_pred,
            "model_input": prompt
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
        print(f"Resume mode: {len(processed_indices)} samples processed, {len(test_dataset) - len(processed_indices)} samples remaining")
    
    # Filter out processed samples
    remaining_dataset = []
    remaining_indices = []
    for i, sample in enumerate(test_dataset):
        if i not in processed_indices:
            remaining_dataset.append(sample)
            remaining_indices.append(i)
    
    if not remaining_dataset:
        print("All samples processed, no inference needed")
        return count_all_predictions(output_jsonl, len(test_dataset))
    
    print(f"Multi-threaded inference, threads: {max_workers}, image indices per sample: {num_images_idxs}, samples to process: {len(remaining_dataset)}")
    # Use original indices (i) instead of indices in remaining_dataset
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
    print(f"Inference results saved to: {output_jsonl}, total {len(all_results)} entries (new: {len(results)})")
    
    # Return number of processed samples (including error results)
    return count_all_predictions(output_jsonl, len(test_dataset))

def parse_api_keys_and_setup_manager(api_key, rpm=None, infer=False, default_rpm=20):
    """
    Parse API keys and RPM parameters, and setup APIKeyManager.
    
    Args:
        api_key: API key, can be string, list, etc., supports comma-separated or Python list string
        rpm: RPM limit, can be string, list, integer or None, supports comma-separated or Python list string
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
        num_images_idxs: can be None, string (e.g. "[0,2,3]" or "0,1,2"), etc.
    
    Returns:
        list[int]: image index list
    """
    import ast
    if num_images_idxs is None:
        return [0]
    else:
        try:
            # Support Python list string, e.g. "[0,2,3]"
            num_images_idxs = ast.literal_eval(num_images_idxs)
            if isinstance(num_images_idxs, int):
                num_images_idxs = [num_images_idxs]
            elif not isinstance(num_images_idxs, list):
                num_images_idxs = [int(num_images_idxs)]
            else:
                num_images_idxs = [int(x) for x in num_images_idxs]
        except Exception:
            # e.g. "0,1,2"
            num_images_idxs = [int(x) for x in num_images_idxs.split(",") if str(x).strip() != '']
    return num_images_idxs

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Script start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action='store_true', default=False, help="Whether to perform inference")
    parser.add_argument("--evaluate", action='store_true', default=False, help="Whether to evaluate")
    parser.add_argument("--api_url", type=str, default="http://v1", help="Inference API URL")
    parser.add_argument("--api_key", type=str, default="test", help="API KEY/Multiple API KEYs supported, comma-separated or Python list/list string")
    parser.add_argument("--rpm", type=str, default=None, help="RPM for each api_key, e.g.: 20 or [20,15,30] or 20,20,20")
    parser.add_argument("--test_jsonl_path", type=str, default='', help="Test set jsonl path (English stage1 dataset)")
    parser.add_argument("--output_pred_jsonl", type=str, default="./evaluate_results/output_pred.jsonl", help="Inference output jsonl path")
    parser.add_argument("--sample_num", type=int, default=10000, help="Number of samples for inference or evaluation")
    parser.add_argument("--model_name", type=str, default="judge", help="Model name")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of concurrent threads")
    parser.add_argument("--num_images_idxs", type=str, default='0', help="List of image indices, e.g. [0,1] or 0,1, indicating image idx used for each input (0 is standard image, 1+ are user images). Default uses only standard image.")
    parser.add_argument("--resume", action='store_true', default=False, help="Whether to resume from existing output file, skip already processed samples")
    parser.add_argument("--max_retry_loops", type=int, default=5, help="Maximum number of loops, if inference fails to process all samples, automatically resume and infer again until max loops reached or all samples processed")
    args = parser.parse_args()
    # Process image index list
    num_images_idxs = parse_num_images_idxs(args.num_images_idxs)

    test_dataset = load_test_dataset(args.test_jsonl_path)
    # New dataset is already filtered, no need to filter again
    if args.sample_num:
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
                print(f"\nAll samples processed! Total: {processed_count}/{total_samples} samples")
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
        evaluate_predictions(test_dataset, args.output_pred_jsonl)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Script end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_time}")
