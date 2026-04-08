import json
import os
import re
import ast
import time
from typing import Dict, Any, Tuple, List, Optional
from openai import OpenAI

JUDGE_SCORE_KEYS = {"T_score", "N_score", "M_score"}
MAX_JUDGE_ATTEMPTS = 3

# API Configurations
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "your-api-key")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")

def make_judge_client() -> OpenAI:
    return OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    def _parse_obj(candidate: str) -> Optional[Dict[str, Any]]:
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _balanced_json_objects(s: str) -> List[str]:
        objs: List[str] = []
        start = -1
        depth = 0
        in_str = False
        quote = ""
        escape = False
        for i, ch in enumerate(s):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue

            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue

            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objs.append(s[start : i + 1])
                    start = -1
        return objs

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block:
        parsed = _parse_obj(code_block.group(1))
        if parsed:
            return parsed

    for candidate in _balanced_json_objects(cleaned):
        parsed = _parse_obj(candidate)
        if parsed:
            return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _parse_obj(cleaned[start : end + 1])
        if parsed:
            return parsed

    return None

def validate_judge_json(judge_json: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(judge_json, dict):
        return False, "top-level is not an object"
    if "scores" not in judge_json or "justification" not in judge_json:
        return False, "judge JSON must contain scores and justification"

    scores = judge_json.get("scores")
    if not isinstance(scores, dict):
        return False, "scores must be an object"

    missing_scores = JUDGE_SCORE_KEYS - set(scores.keys())
    if missing_scores:
        return False, f"missing score keys: {sorted(missing_scores)}"

    for key in JUDGE_SCORE_KEYS:
        value = scores.get(key)
        if not isinstance(value, int) or not (1 <= value <= 5):
            return False, f"{key} must be int in 1..5"

    justification = judge_json.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        if not isinstance(justification, dict):
            return False, "justification must be a dictionary or a valid string"
    return True, "ok"

def normalize_judge_json(judge_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(judge_json, dict):
        return {}

    out = dict(judge_json)
    scores = out.get("scores") if isinstance(out.get("scores"), dict) else {}
    norm_scores: Dict[str, int] = {}
    for key in JUDGE_SCORE_KEYS:
        value = scores.get(key)
        if isinstance(value, int):
            norm_scores[key] = max(1, min(5, value))
    out["scores"] = norm_scores

    justification = out.get("justification")
    if isinstance(justification, str):
        out["justification"] = justification.strip()
    elif isinstance(justification, (dict, list)):
        out["justification"] = json.dumps(justification, ensure_ascii=False)
    elif justification is None:
        out["justification"] = ""
    else:
        out["justification"] = str(justification).strip()

    return out

def build_judge_messages(ocr_text: str, gt_simplified: Dict[str, str], pred_simplified: Dict[str, Any]) -> List[Dict[str, Any]]:
    gt_simplified_json = json.dumps(gt_simplified, ensure_ascii=False, indent=2)
    pred_simplified_json = json.dumps(pred_simplified, ensure_ascii=False, indent=2)

    judge_prompt = f"""
你是肺癌 TNM benchmark 的严格评测裁判。请只基于以下 GT_Simplified、Pred_Simplified 以及 AJCC 第八版肺癌 TNM 分期规则进行评估；下面提供的检查报告OCR文本仅用于核对是否存在明显与报告相矛盾的情况，但评分以 GT_Simplified 和 Pred_Simplified 为主。

检查报告内容 (OCR Text):
```
{ocr_text}
```

GT_Simplified:
{gt_simplified_json}

Pred_Simplified:
{pred_simplified_json}

你的任务：
分别对 T、N、M 三个维度独立打分，每项 1-5 分。
请重点评估两件事：
1. Pred 的 stage 是否与 GT 一致。
2. Pred 的 reasoning 是否符合该维度的关键证据逻辑与工作流要求。

总原则：
- stage 错误必须明显扣分。
- stage 正确但 reasoning 不符合该维度规则，不能给 5 分。
- 不得因为整体看起来“差不多”而放松对单维度规则的要求。
- 不得把 GT 中不存在的证据说成 Pred 正确引用的依据。
- 若 Pred 把 uncertain / 待定信息当作 confirmed 事实，至少降 1 档。
- 输出必须是合法 JSON，不要输出任何额外文字。

--------------------------------
一、T_score 评分规则
--------------------------------
评估重点：
A. T_stage 是否与 GT 完全一致。
B. T_reasoning 是否抓住决定 T 分期的关键证据。
C. T_reasoning 是否符合以下规则：
   - 应基于标签级证据表达，不要因为 Pred 复述精确测量值/SUV 就给高分；若 reasoning 主要依赖精确数值复述而缺少标签级结论，需降分。
   - 如果病例同时满足多个 T 标准，reasoning 应体现“先识别全部已满足标准，再按最 高级别确定 final T_stage”的思路。
   - 不应只说大小而漏掉更高级别的侵犯/播散证据。
   - 不应把属于 M 的证据（如胸膜转移、胸壁转移、对侧肺转移）误当成 T 依据。
   - 若 GT 的 T_stage 是由同侧同肺叶播散 / 同侧不同肺叶播散 / 癌性淋巴管炎 / 邻 近结构侵犯等决定，reasoning 必须明确提到该决定性证据，而不只是泛泛说“肿瘤较大”。

T_score 细则：
- 5分：T_stage 完全正确；T_reasoning 命中决定性依据，体现最高级别原则，无明显误 用证据、过度推断或格式性缺陷。
- 4分：T_stage 正确；T_reasoning 基本正确，但略有不完整，例如漏掉一个次要已满足 标准，或表达偏弱但仍抓住决定性依据。
- 3分：T_stage 正确但 reasoning 明显薄弱，或 stage 只差一个子级且 reasoning 有部分合理依据。
- 2分：T_stage 错误，但 reasoning 仍有部分相关证据或方向接近。
- 1分：T_stage 严重错误，或 reasoning 存在明显幻觉、关键证据错配、把 M 证据当 T 证据等严重问题。

--------------------------------
二、N_score 评分规则
--------------------------------
评估重点：
A. N_stage 是否与 GT 完全一致。
B. N_reasoning 是否符合“全量扫描 + 最高 confirmed stage 落点”的规则。
C. N_reasoning 是否符合以下规则：
   - 应体现对 N1/N2/N3 证据的扫描，而不是只提一个节点就直接跳结论。
   - reasoning 最终必须明确落到最终 N_stage（例如“因此最高已确认淋巴结分期为 N2”）。
   - 不应把 uncertain / 性质待定 / 稍大但未明确转移 的节点计入最终 confirmed N_stage。
   - 若存在 uncertain node，可以讨论其潜在升级方向，但 final stage 必须基于 confirmed evidence。
   - 不应把非区域淋巴结误算为 N 分期依据。

N_score 细则：
- 5分：N_stage 完全正确；reasoning 明确呈现关键淋巴结证据，并明确落到最终最高 confirmed N_stage；confirmed 与 uncertain 区分清楚。
- 4分：N_stage 正确；reasoning 基本完整，但全量扫描意识稍弱，或 final landing 不够有力但仍可清楚判断其落点。
- 3分：N_stage 正确但 reasoning 明显不完整，例如只列局部证据、未清楚落到最终 stage；或 stage 只差一级但证据方向基本对。
- 2分：N_stage 错误，但 reasoning 仍提到部分相关淋巴结证据。
- 1分：N_stage 严重错误，或将 uncertain / 非区域淋巴结 / 错误侧别直接当作 confirmed N 依据。

--------------------------------
三、M_score 评分规则
--------------------------------
评估重点：
A. M_stage 是否与 GT 完全一致。
B. M_reasoning 是否符合 M1a / M1b / M1c 的判定路径。
C. M_reasoning 是否符合以下规则：
   - 应先识别 M1a 相关证据：对侧肺、胸膜转移、恶性胸腔积液、恶性心包积液等。
   - 应明确区分肺外器官/非区域淋巴结的器官数与灶数。
   - 单器官单灶应对应 M1b；单器官多灶或多器官应对应 M1c。
   - 不应把区域淋巴结误当作 M 证据。
   - 不应把“待定”“随诊观察”“仅肿大/代谢增高但无转移表述”的病灶直接算作 confirmed metastasis。

M_score 细则：
- 5分：M_stage 完全正确；M_reasoning 明确说明决定该 stage 的路径（M0 / M1a / M1b / M1c），器官数/灶数逻辑正确，无明显误判。
- 4分：M_stage 正确；reasoning 基本正确，但对判定路径说明不够完整，或遗漏次要伴 随证据。
- 3分：M_stage 正确但 reasoning 较弱，或 stage 接近正确但对 M1b/M1c 的逻辑表达不清。
- 2分：M_stage 错误，但 reasoning 中仍包含部分相关远处转移证据。
- 1分：M_stage 严重错误，或把区域淋巴结/待定病灶/无转移表述错误算作 confirmed metastasis。

--------------------------------
四、输出要求
--------------------------------
只输出一个 JSON 对象，格式如下：

{
    "scores": {
        "T_score": 0,
        "N_score": 0,
        "M_score": 0
    },
    "justification": {
        "T": "说明 T 为什么给这个分，指出 stage 是否一致、关键证据是否命中、主要扣分点。",
        "N": "说明 N 为什么给这个分，指出是否明确落到最终 stage、是否错误使用 uncertain 或非区域节点。",
        "M": "说明 M 为什么给这个分，指出是否正确区分 M0/M1a/M1b/M1c，以及主要扣分点。"
    }
}
"""
    return [
        {
            "role": "system",
            "content": "你是严格客观的医学评测裁判，只输出合法 JSON。",
        },
        {"role": "user", "content": judge_prompt},
    ]

def call_judge_until_valid(client: OpenAI, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content
            parsed = extract_json(raw if isinstance(raw, str) else str(raw))
            if not parsed:
                last_error = "judge output is not valid JSON"
                continue
            parsed = normalize_judge_json(parsed)
            ok, err = validate_judge_json(parsed)
            if ok:
                return parsed
            last_error = err
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_JUDGE_ATTEMPTS:
            time.sleep(2)

    return {
        "error": f"Judge failed after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}",
        "scores": {"T_score": 1, "N_score": 1, "M_score": 1},
        "justification": str(last_error)
    }

def process_results_with_judge(results_dir: str, gt_file: str):
    """
    Apply the exact judge workflow over evaluated JSONs.
    """
    client = make_judge_client()
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    gt_dict = {item.get("id"): item for item in ground_truth}
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'): continue
        file_path = os.path.join(results_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            
        print(f"\nEvaluating: {filename}...")
        
        total = 0
        perfect_TNMs = 0
        
        for pred in predictions:
            case_id = pred.get("id")
            if case_id not in gt_dict: continue
                
            gt_data = gt_dict[case_id]
            ocr_text = gt_data.get("clinical_text", gt_data.get("ocr_text", ""))
            gt_simplified = { 
                "T_stage": gt_data.get("T_stage", ""), "N_stage": gt_data.get("N_stage", ""), "M_stage": gt_data.get("M_stage", "") 
            }
            pred_val = pred.get("response", pred)
            
            messages = build_judge_messages(ocr_text, gt_simplified, pred_val)
            eval_res = call_judge_until_valid(client, messages)
            
            scores = eval_res.get("scores", {})
            t_score = scores.get("T_score", 0)
            n_score = scores.get("N_score", 0)
            m_score = scores.get("M_score", 0)
            
            total += 1
            if t_score >= 4 and n_score >= 4 and m_score >= 4:
                perfect_TNMs += 1
            
            print(f"[{case_id}] T:{t_score} N:{n_score} M:{m_score}")
            
        acc = (perfect_TNMs / total * 100) if total > 0 else 0
        print(f"--> {filename} | Strict Match (Scores >= 4): {acc:.2f}% ({perfect_TNMs}/{total})")

if __name__ == "__main__":
    print("Agentic Judge evaluation initiated.")
    # Example usage:
    # process_results_with_judge("../results_TNMstaging/results_Agent", "../data/ground_truth.json")
