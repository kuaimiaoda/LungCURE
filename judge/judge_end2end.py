import os
import json
import yaml
import requests
import argparse
import re


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    judge_provider_name = config['evaluation']['judge_provider']
    return config['providers'][judge_provider_name]


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _call_llm(judge_config, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    """调用裁判模型，失败自动重试，超过次数后抛出异常"""
    url = f"{judge_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {judge_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": judge_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": judge_config.get("temperature", 0.2),
        "max_tokens": judge_config.get("max_tokens", 2048),
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            print(f"  [Retry {attempt}/{max_retries}] API 调用失败: {e}")
    raise RuntimeError(f"API 调用连续失败 {max_retries} 次，最后错误: {last_error}")


def compare_medications_via_llm(gt_text, pred_text, judge_config, language="Chinese"):
    """提取并语义匹配药物列表"""
    if language.lower().startswith("english"):
        system_prompt = "You are a medical expert specializing in oncology drug evaluation. You must output strict JSON only."
        user_prompt = f"""Please complete the following tasks:

1. Extract all recommended medications (including chemotherapy, targeted therapy, immunotherapy, bone protection drugs, etc.) from the "Ground Truth Text", using standard English generic names.
2. Extract all recommended medications from the "Prediction Text", using standard English generic names.
3. Determine which medications from the two lists are the same. Consider different expressions of the same drug, for example:
   - Brand name vs generic name: Taxol = paclitaxel = PTX
   - Abbreviations: DDP = cisplatin = cis-platinum
   - Salt form differences: carboplatin = carbo

Output strictly in the following JSON format, without any explanatory text or Markdown markers (e.g. ```json):
{{
    "gt_medications": ["drug1", "drug2"],
    "pred_medications": ["drugA", "drugB"],
    "matched_count": 2,
    "matched_pairs": [["name in gt", "name in pred"]]
}}

Ground Truth Text:
{gt_text}

Prediction Text:
{pred_text}"""
    else:
        system_prompt = "你是医疗领域的药物评估专家。必须严格输出 JSON，不要输出其他内容。"
        user_prompt = f"""请完成以下任务：

1. 从"Ground Truth 文本"中提取所有建议使用的药物（包含化疗药、靶向药、免疫药、骨保护药等），统一使用中文通用名。
2. 从"预测文本"中提取所有建议使用的药物，统一使用中文通用名。
3. 判断两个列表中哪些药物是相同的。判断时需考虑同一药物的不同表达方式，例如：
   - 中英文互换：顺铂 = cisplatin = DDP
   - 缩写：表柔比星 = epirubicin = EPI
   - 商品名与通用名：泰素 = 紫杉醇 = paclitaxel = PTX
   - 盐型差异：盐酸表柔比星 = 表柔比星

严格按照以下JSON格式输出，不要包含任何说明文字或Markdown标记（如```json）：
{{
    "gt_medications": ["药物1", "药物2"],
    "pred_medications": ["药物A", "药物B"],
    "matched_count": 2,
    "matched_pairs": [["gt中的名称", "pred中的名称"]]
}}

Ground Truth 文本：
{gt_text}

预测文本：
{pred_text}"""

    result_content = _call_llm(judge_config, system_prompt, user_prompt)
    data = _extract_json_object(result_content)
    gt_meds = data.get("gt_medications", [])
    pred_meds = data.get("pred_medications", [])
    matched_pairs = data.get("matched_pairs", [])
    return gt_meds, pred_meds, matched_pairs


def judge_cdss_accuracy(gt_text, pred_text, judge_config, language="Chinese", case_id=""):
    """对 CDSS 结果相似度打分（0-5），返回归一化分数(0-1)和理由"""
    if not gt_text or not pred_text:
        reason = (
            "Missing predicted or GT cdss_result; unable to evaluate similarity."
            if language.lower().startswith("english")
            else "缺少预测或GT的cdss_result，无法进行有效比对。"
        )
        return 0.0, reason

    if language.lower().startswith("english"):
        system_prompt = "You are an oncology clinical reviewer. You must output strict JSON only, with no extra text."
        user_prompt = f"""Please evaluate CDSS result similarity. Case ID: {case_id}

[Base model CDSS result]
{pred_text}

[GT CDSS result]
{gt_text}

Score only from the perspective of result similarity, and do not evaluate writing style.
- score: integer from 0 to 5
- reason: brief rationale (within 120 words)

Output JSON only:
{{
  "score": 0,
  "reason": ""
}}"""
    else:
        system_prompt = "你是肿瘤临床评审专家。必须严格输出 JSON，不要输出其他内容。"
        user_prompt = f"""请评估 CDSS 结果相似度。病例ID: {case_id}

【基座模型 CDSS 结果】
{pred_text}

【GT CDSS 结果】
{gt_text}

只从"结果相似度"角度打分，不评估风格。
- score: 0-5 整数
- reason: 简短理由（不超过120字）

只输出 JSON：
{{
  "score": 0,
  "reason": ""
}}"""

    raw = _call_llm(judge_config, system_prompt, user_prompt)
    parsed = _extract_json_object(raw)
    raw_score = parsed.get("score", 0)
    score_int = max(0, min(5, int(round(float(raw_score)))))
    reason = str(parsed.get("reason", "")).strip()
    return round(score_int / 5.0, 4), reason


def calculate_score(gt_meds, pred_meds, matched_pairs):
    """计算得分: 匹配药物数 / 预测药物数"""
    if len(pred_meds) == 0:
        return 1.0 if len(gt_meds) == 0 else 0.0
    return len(matched_pairs) / len(pred_meds)


def extract_cases(pred_data) -> dict:
    """从预测数据中提取 cases 字典，兼容以下格式：
    - dict with 'cases' key: {"cases": {"case_id": {...}}}  (kimi/llm_batch 格式)
    - flat dict keyed by case_id: {"LC_patient_0007": {...}}
    - list of case objects: [{"case_id": "...", ...}]       (test_ge 格式)
    """
    if isinstance(pred_data, list):
        return {
            str(item["case_id"]): item
            for item in pred_data
            if isinstance(item, dict) and "case_id" in item
        }
    if isinstance(pred_data, dict):
        cases = pred_data.get("cases")
        if isinstance(cases, dict):
            return cases
        # flat dict: 值为 dict 且含 case_id 字段
        first = next(iter(pred_data.values()), None)
        if isinstance(first, dict) and "case_id" in first:
            return pred_data
    return {}


def extract_cdss_result(case_obj: dict) -> str:
    """从 case 对象中提取 cdss_result，支持直接字段或嵌套在 tnm_result 中"""
    if not isinstance(case_obj, dict):
        return ""
    cdss = str(case_obj.get("cdss_result") or "").strip()
    if cdss:
        return cdss
    tnm = case_obj.get("tnm_result")
    if isinstance(tnm, dict):
        return str(tnm.get("cdss_result") or "").strip()
    return ""


def find_gt_file(pred_filename, gt_dir):
    """匹配 GT 文件"""
    match = re.search(r'_([A-Za-z]+)_seed(\d+)', pred_filename)
    if not match:
        return None

    language = match.group(1)
    seed = match.group(2)

    if not os.path.exists(gt_dir):
        return None

    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith('.json') and f"_{language}_" in gt_file and f"seed{seed}" in gt_file:
            return os.path.join(gt_dir, gt_file)

    return None


def build_metric_filename(pred_filename: str) -> str:
    """将输入文件名 [model]_output_[chinese/english]_[seedN].json
    转为输出文件名 [model]_metric_[chinese/english]_[seedN].json。
    """
    m = re.match(
        r"^(?P<model>.+)_output_(?P<lang>chinese|english)_(?P<seed>seed\d+)\.json$",
        pred_filename,
        flags=re.IGNORECASE,
    )
    if m:
        model = m.group("model")
        lang = m.group("lang")
        seed = m.group("seed")
        return f"{model}_metric_{lang}_{seed}.json"

    # 兜底：保持原行为，避免异常命名导致中断。
    base_name = os.path.splitext(pred_filename)[0]
    return f"{base_name}_metrics.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM medication predictions against dynamically matched GT files.")
    parser.add_argument("--config", type=str, default="api_config.yaml", help="Path to api_config.yaml")
    parser.add_argument("--gt_dir", type=str, default="gt/cdss_gt", help="Directory containing GT JSON files")
    parser.add_argument("--input_dir", type=str, default="results/outputs/llm_outputs/llm_text_output/gpt-5.2", help="Directory containing prediction JSONs")
    parser.add_argument("--output_dir", type=str, default="results/metrics/llm_metrics/llm_text_metrics", help="Directory to save the output JSON metrics")
    parser.add_argument("--max-file", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases per file to process")

    args = parser.parse_args()

    judge_config = load_config(args.config)

    if not os.path.exists(args.input_dir):
        print(f"Error: 输入文件夹不存在 -> {args.input_dir}")
        return
    if not os.path.exists(args.gt_dir):
        print(f"Error: GT文件夹不存在 -> {args.gt_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    if args.max_file is not None:
        json_files = json_files[:args.max_file]

    gt_cache = {}

    for file_name in json_files:
        print(f"\nProcessing file: {file_name}")

        gt_file_path = find_gt_file(file_name, args.gt_dir)
        if not gt_file_path:
            print(f"  [Warning] 在 {args.gt_dir} 中未找到与 {file_name} 匹配的 GT 文件，跳过。")
            continue

        print(f"  Matched GT file: {os.path.basename(gt_file_path)}")
        lang_match = re.search(r'_([A-Za-z]+)_seed', file_name)
        language = lang_match.group(1) if lang_match else "Chinese"

        if gt_file_path not in gt_cache:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_cache[gt_file_path] = json.load(f)
        gt_data = gt_cache[gt_file_path]

        file_path = os.path.join(args.input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        cases = extract_cases(pred_data)
        case_ids = list(cases.keys())

        if args.max_cases is not None:
            case_ids = case_ids[:args.max_cases]

        case_results = {}

        skipped_cases = []
        for case_id in case_ids:
            print(f"  Evaluating case: {case_id}")
            if case_id not in gt_data:
                print(f"  [Warning] Case {case_id} 未在GT文件中找到，跳过。")
                continue

            gt_text = extract_cdss_result(gt_data[case_id])
            pred_text = extract_cdss_result(cases[case_id])

            try:
                # 药物匹配得分
                gt_meds, pred_meds, matched_pairs = compare_medications_via_llm(gt_text, pred_text, judge_config, language)
                score = calculate_score(gt_meds, pred_meds, matched_pairs)

                # CDSS 相似度 F1 得分
                f1_score, f1_reason = judge_cdss_accuracy(gt_text, pred_text, judge_config, language, case_id)
            except Exception as e:
                print(f"  [Skip] Case {case_id} 重试3次仍失败，已跳过。原因: {e}")
                skipped_cases.append(case_id)
                continue

            case_results[case_id] = {
                "gt_medications": gt_meds,
                "pred_medications": pred_meds,
                "matched_pairs": matched_pairs,
                "score": score,
                "f1_score": f1_score,
                "f1_reason": f1_reason,
            }

        # 计算平均分
        scores = [v["score"] for v in case_results.values()]
        f1_scores = [v["f1_score"] for v in case_results.values()]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        avg_f1_score = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

        # 汇总信息
        output = {
            "average_score": avg_score,
            "average_f1_score": avg_f1_score,
            "case_count": len(case_results),
            "skipped_cases": skipped_cases,
        } | case_results

        metric_file_name = build_metric_filename(file_name)
        output_path = os.path.join(args.output_dir, metric_file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ {file_name} finished！average_score={avg_score}, average_f1_score={avg_f1_score} -> {output_path}")

    print(f"\n🎉 全部完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
