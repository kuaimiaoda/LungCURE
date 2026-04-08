import os
import json
import yaml
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.common.model_api import chat_content_with_retry


def load_config(config_path):
    """加载裁判模型配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    judge_provider_name = config['evaluation']['judge_provider']
    return config['providers'][judge_provider_name]


def _extract_json_object(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象"""
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


def _call_llm(judge_config, system_prompt: str, user_prompt: str) -> str:
    """统一调用裁判模型"""
    api_key = str(judge_config.get("api_key", "")).strip()
    base_url = str(judge_config.get("base_url", "")).strip()
    model = str(judge_config.get("model", "")).strip()
    if not api_key or not base_url or not model:
        raise ValueError("judge_config 缺少 api_key/base_url/model")

    return chat_content_with_retry(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=float(judge_config.get("temperature", 0.2)),
        max_tokens=int(judge_config.get("max_tokens", 2048)),
        timeout=int(judge_config.get("timeout", 60)),
        max_retries=int(judge_config.get("max_retries", 3)),
        retry_backoff_seconds=float(judge_config.get("retry_backoff_seconds", 2.0)),
        alias_retry=True,
    )
def compare_medications_via_llm(gt_text, pred_text, judge_config, language="Chinese"):
    """调用裁判大模型提取并语义匹配药物列表"""
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

    try:
        result_content = _call_llm(judge_config, system_prompt, user_prompt)
        data = _extract_json_object(result_content)
        gt_meds = data.get("gt_medications", [])
        pred_meds = data.get("pred_medications", [])
        matched_pairs = data.get("matched_pairs", [])
        return gt_meds, pred_meds, matched_pairs
    except Exception as e:
        print(f"  [Error] 调用裁判模型或解析JSON失败: {e}")
        return [], [], []


def judge_cdss_accuracy(gt_text, pred_text, judge_config, language="Chinese", case_id=""):
    """调用裁判模型对 CDSS 结果相似度打分（0-5），返回归一化分数(0-1)和理由"""
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

    try:
        raw = _call_llm(judge_config, system_prompt, user_prompt)
        parsed = _extract_json_object(raw)
        raw_score = parsed.get("score", 0)
        score_int = max(0, min(5, int(round(float(raw_score)))))
        reason = str(parsed.get("reason", "")).strip()
        return round(score_int / 5.0, 4), reason
    except Exception as e:
        print(f"  [Error] judge_cdss_accuracy 调用失败: {e}")
        return 0.0, f"error: {e}"


def calculate_score(gt_meds, pred_meds, matched_pairs):
    """计算得分: 匹配药物数 / 预测药物数"""
    if len(pred_meds) == 0:
        return 1.0 if len(gt_meds) == 0 else 0.0
    return len(matched_pairs) / len(pred_meds)


def find_gt_file(pred_filename, gt_dir):
    """根据预测文件名中的 language 和 seed 匹配对应的 GT 文件"""
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


def _first_non_empty_text(item: dict, keys: list[str]) -> str:
    for k in keys:
        v = str(item.get(k, "")).strip()
        if v:
            return v
    return ""


def _extract_cases(payload, *, is_pred: bool) -> dict:
    """
    兼容多种输入：
    1) list[case_dict]
    2) {"cases": {...}}
    3) {case_id: case_dict}
    """
    result = {}

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            cid = (
                str(item.get("case_id", "")).strip()
                or str(item.get("id", "")).strip()
                or str(item.get("病例ID", "")).strip()
            )
            if not cid:
                continue
            result[cid] = item
        return result

    if not isinstance(payload, dict):
        return result

    raw_cases = payload.get("cases")
    if isinstance(raw_cases, dict):
        for key, item in raw_cases.items():
            if not isinstance(item, dict):
                continue
            cid = (
                str(item.get("case_id", "")).strip()
                or str(item.get("id", "")).strip()
                or str(item.get("病例ID", "")).strip()
                or str(key).strip()
            )
            if cid:
                result[cid] = item
        return result

    # 顶层就是 case_id -> case_obj
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        cid = (
            str(item.get("case_id", "")).strip()
            or str(item.get("id", "")).strip()
            or str(item.get("病例ID", "")).strip()
            or str(key).strip()
        )
        if cid:
            result[cid] = item

    # 预测数据若无可用病例，尝试把顶层对象视作单病例
    if is_pred and not result and payload:
        cid = (
            str(payload.get("case_id", "")).strip()
            or str(payload.get("id", "")).strip()
            or str(payload.get("病例ID", "")).strip()
        )
        if cid:
            result[cid] = payload

    return result


def _extract_pred_cdss_text(case_obj: dict) -> str:
    # 兼容你的输入结构：cdss_result 可能为空，主要在 treatment/final_answer
    text = _first_non_empty_text(case_obj, ["cdss_result", "treatment", "final_answer"])
    if text:
        return text

    # 兼容部分结构化输出
    base_model_cdss = case_obj.get("base_model_cdss")
    if isinstance(base_model_cdss, dict):
        text = _first_non_empty_text(base_model_cdss, ["cdss_result", "treatment", "final_answer"])
        if text:
            return text
    return ""


def _extract_gt_cdss_text(case_obj: dict) -> str:
    return _first_non_empty_text(case_obj, ["cdss_result", "treatment", "final_answer"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM medication predictions against dynamically matched GT files.")
    parser.add_argument("--config", type=str, default="api_config.yaml", help="Path to api_config.yaml")
    parser.add_argument("--gt_dir", type=str, default="cdss_gt", help="Directory containing GT JSON files")
    parser.add_argument("--input_dir", type=str, default="llm_batch/llm_batch/text_results/kimi-k2.5/outputs", help="Directory containing prediction JSONs")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save the output JSON metrics")
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
        gt_cases = _extract_cases(gt_data, is_pred=False)

        file_path = os.path.join(args.input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        cases = _extract_cases(pred_data, is_pred=True)
        case_ids = list(cases.keys())

        if args.max_cases is not None:
            case_ids = case_ids[:args.max_cases]

        case_results = {}

        for case_id in case_ids:
            print(f"  Evaluating case: {case_id}")
            if case_id not in gt_cases:
                print(f"  [Warning] Case {case_id} 未在GT文件中找到，跳过。")
                continue

            gt_text = _extract_gt_cdss_text(gt_cases[case_id])
            pred_text = _extract_pred_cdss_text(cases[case_id])

            # 药物匹配得分
            gt_meds, pred_meds, matched_pairs = compare_medications_via_llm(gt_text, pred_text, judge_config, language)
            score = calculate_score(gt_meds, pred_meds, matched_pairs)

            # CDSS 相似度 F1 得分
            f1_score, f1_reason = judge_cdss_accuracy(gt_text, pred_text, judge_config, language, case_id)

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

        # 汇总信息写在最上方
        output = {
            "average_score": avg_score,
            "average_f1_score": avg_f1_score,
            "case_count": len(case_results),
        } | case_results

        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_metrics.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ {file_name} 评估完成！average_score={avg_score}, average_f1_score={avg_f1_score} -> {output_path}")

    print(f"\n🎉 所有评估任务完成！整体输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

