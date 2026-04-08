import os
import argparse
import json
import traceback
import re
from tqdm import tqdm
from eval import cal_em, cal_f1
try:
    from eval_r import cal_rsim
except Exception:
    cal_rsim = None
from eval_g import cal_gen
from concurrent.futures import ThreadPoolExecutor


def evaluate_one(d):
    try:
        generation = d['generation']
        try:
            answer = generation.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            answer = generation
        em_score = cal_em([d['golden_answers']], [answer])
        f1_score = cal_f1([d['golden_answers']], [answer])

        # 去重 context
        context = []
        for c in d['context']:
            if c not in context:
                context.append(c)

        rsim_score = cal_rsim(['\n'.join(context)], [d['knowledge']]) if (cal_rsim is not None and d['knowledge'] != "") else 0.0 
        gen_score = cal_gen(d['question'], d['golden_answers'], generation, f1_score)

        d['em'] = em_score
        d['f1'] = f1_score
        d['rsim'] = rsim_score
        d['gen'] = gen_score["score"]
        d['gen_exp'] = gen_score["explanation"]

        return d
    except Exception as e:
        print(f"[ERROR] Failed processing sample: {d.get('question', 'N/A')}")
        traceback.print_exc()
        raise


def _extract_cases(payload):
    if not isinstance(payload, dict):
        return {}
    cases = payload.get("cases")
    if isinstance(cases, dict):
        return cases
    return payload


def _extract_language_seed(file_name):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    tokens = [t.strip() for t in stem.split("_") if t.strip()]
    language = ""
    seed = ""

    for token in tokens:
        lower = token.lower()
        if lower == "chinese":
            language = "Chinese"
        elif lower == "english":
            language = "English"

        m = re.fullmatch(r"seed(\d+)", lower)
        if m:
            seed = m.group(1)

    if not language or not seed:
        raise ValueError(f"Cannot parse language/seed from file name: {file_name}")
    return language, seed


def _resolve_gt_file(pred_file, gt_dir):
    language, seed = _extract_language_seed(pred_file)
    gt_name = f"benchmark_gt_cdss_{language}_seed{seed}.json"
    gt_path = os.path.join(gt_dir, gt_name)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found for {pred_file}: {gt_path}")
    return gt_path


def _extract_cdss_result(case_obj):
    if not isinstance(case_obj, dict):
        return ""
    root = str(case_obj.get("cdss_result", "")).strip()
    if root:
        return root
    tnm_result = case_obj.get("tnm_result")
    if isinstance(tnm_result, dict):
        return str(tnm_result.get("cdss_result", "")).strip()
    return ""


def _build_question(language):
    if str(language).lower().startswith("english"):
        return "Generate a concise CDSS treatment recommendation for a lung cancer patient based on case information."
    return "根据患者病例信息，生成简洁且可执行的肺癌CDSS治疗建议。"


def _prepare_test_generation_from_dirs(pred_dir, gt_dir, max_files=0, max_cases=0):
    pred_files = sorted(
        [
            os.path.join(pred_dir, f)
            for f in os.listdir(pred_dir)
            if f.lower().endswith(".json") and os.path.isfile(os.path.join(pred_dir, f))
        ]
    )
    if max_files > 0:
        pred_files = pred_files[:max_files]

    if not pred_files:
        raise FileNotFoundError(f"No json files found in pred_dir: {pred_dir}")

    data = []
    for pred_file in pred_files:
        gt_file = _resolve_gt_file(pred_file, gt_dir)
        language, _ = _extract_language_seed(pred_file)
        question = _build_question(language)

        with open(pred_file, "r", encoding="utf-8-sig") as f:
            pred_payload = json.load(f)
        with open(gt_file, "r", encoding="utf-8-sig") as f:
            gt_payload = json.load(f)

        pred_cases = _extract_cases(pred_payload)
        gt_cases = _extract_cases(gt_payload)

        case_items = list(pred_cases.items())
        if max_cases > 0:
            case_items = case_items[:max_cases]

        for case_id, pred_case in case_items:
            case_key = str(case_id).strip()
            gt_case = gt_cases.get(case_key, {}) if isinstance(gt_cases, dict) else {}

            generation = _extract_cdss_result(pred_case)
            golden_answer = _extract_cdss_result(gt_case)

            data.append(
                {
                    "question": question,
                    "golden_answers": [golden_answer],
                    "generation": generation,
                    "context": [],
                    "knowledge": "",
                    "meta": {
                        "pred_file": pred_file,
                        "gt_file": gt_file,
                        "case_id": case_key,
                    },
                }
            )

    return data

def evaluate_method(args):
    method = args.method
    data_source = args.data_source
    success_flag = False  # 控制是否成功保存

    try:
        # print(f"[DEBUG] Evaluating {method} on {data_source}")
        data_dir = f"results/{method}/{data_source}/test_generation.json"

        if args.api_config:
            os.environ["API_CONFIG_PATH"] = args.api_config

        if args.pred_dir and args.gt_dir:
            data = _prepare_test_generation_from_dirs(
                pred_dir=args.pred_dir,
                gt_dir=args.gt_dir,
                max_files=args.max_files,
                max_cases=args.max_cases,
            )
            os.makedirs(os.path.dirname(data_dir), exist_ok=True)
            with open(data_dir, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[SAVED] {data_dir}")
        else:
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"File not found: {data_dir}")

            with open(data_dir, encoding="utf-8") as f:
                data = json.load(f)

        # 并行处理样本
        max_workers = 64
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data = list(tqdm(executor.map(evaluate_one, data), total=len(data), desc=method))

        # 汇总指标
        overall_em = sum([d['em'] for d in data]) / len(data)
        overall_f1 = sum([d['f1'] for d in data]) / len(data)
        overall_rsim = sum([d['rsim'] for d in data]) / len(data)
        overall_gen = sum([d['gen'] for d in data]) / len(data)

        # print(f"{method} Overall EM: {overall_em:.4f}")
        # print(f"{method} Overall F1: {overall_f1:.4f}")
        # print(f"{method} Overall R-Sim: {overall_rsim:.4f}")
        # print(f"{method} Overall Gen: {overall_gen:.4f}")

        save_base = f"results/{method}/{data_source}"
        os.makedirs(save_base, exist_ok=True)

        result_path = os.path.join(save_base, "test_result.json")
        with open(result_path, 'w') as f:
            json.dump(data, f, indent=4)

        score_path = os.path.join(save_base, "test_score.json")
        with open(score_path, 'w') as f:
            json.dump({
                "overall_em": overall_em,
                "overall_f1": overall_f1,
                "overall_rsim": overall_rsim,
                "overall_gen": overall_gen,
            }, f, indent=4)

        # 成功保存标志
        success_flag = True
        print(f"[SAVED] {result_path}")
        print(f"[SAVED] {score_path}")
        print(f"[SUCCESS] {method} finished and saved.")

    except Exception as e:
        print(f"\n[ERROR] {method} failed due to: {str(e)}")
        traceback.print_exc()
        raise

    if not success_flag:
        raise RuntimeError(f"{method} did not complete saving.")
    
    return True

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--method', type=str, default='HyperGraphRAG_wo_ER')
    parse.add_argument('--data_source', type=str, default='hypertension')
    parse.add_argument('--pred_dir', type=str, default='')
    parse.add_argument('--gt_dir', type=str, default='')
    parse.add_argument('--api_config', type=str, default='api_config.yaml')
    parse.add_argument('--max_files', type=int, default=0)
    parse.add_argument('--max_cases', type=int, default=0)
    args = parse.parse_args()
    evaluate_method(args)