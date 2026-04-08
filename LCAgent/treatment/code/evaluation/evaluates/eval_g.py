import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.common.model_api import UnifiedOpenAIClient, extract_message_content

# API config path can be overridden by environment variable.
# Priority: API_CONFIG_PATH -> ./api_config.yaml -> openai_api_key.txt fallback.
_client = None
_model = "gpt-4o-mini"
_completion_params = {}


def _build_client_from_api_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("api_config.yaml root must be a dict")

    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("api_config.yaml missing providers")

    eval_conf = data.get("evaluation", {})
    judge_name = "judge_model"
    if isinstance(eval_conf, dict):
        judge_name = str(eval_conf.get("judge_provider", "judge_model")).strip() or "judge_model"

    provider = providers.get(judge_name)
    if not isinstance(provider, dict):
        raise ValueError(f"provider not found in api_config.yaml: {judge_name}")

    api_key = str(provider.get("api_key", "")).strip()
    base_url = str(provider.get("base_url", "")).strip()
    model = str(provider.get("model", "")).strip()

    if not api_key or not base_url or not model:
        raise ValueError(f"invalid provider config: {judge_name}")

    return UnifiedOpenAIClient(api_key=api_key, base_url=base_url), model


def _get_client_and_model():
    global _client, _model
    if _client is not None:
        return _client, _model

    config_path = os.environ.get("API_CONFIG_PATH", "").strip() or "api_config.yaml"
    if os.path.exists(config_path):
        try:
            _client, _model = _build_client_from_api_config(config_path)
            return _client, _model
        except Exception as e:
            print(f"[WARN] Failed to load API config from {config_path}: {e}")

    if os.path.exists("openai_api_key.txt"):
        api_key = open("openai_api_key.txt").read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        _client = UnifiedOpenAIClient(api_key=api_key, base_url="https://api.apiyi.com/v1")
        _model = "gpt-4o-mini"
        return _client, _model

    raise FileNotFoundError("No valid API config found. Please provide API_CONFIG_PATH or openai_api_key.txt")

def cal_gen(question, answers, generation, f1_score):
    exp = {}

    def build_prompt(metric):
        descriptions = {
    "comprehensiveness": (
        "comprehensiveness",
        "whether the end-to-end response covers all necessary clinical components, including TNM staging inference, first-line and second-line treatment regimens, and relevant molecular-targeted or immunotherapy options based on driver gene status",
        """Scoring Guide (0–10):
- 10: Fully covers all clinical components from staging inference to treatment lines and molecular considerations.
- 8–9: Covers most key components; only minor omissions in staging or regimen coverage.
- 6–7: Covers some clinical aspects but lacks depth or overlooks notable staging or regimen options.
- 4–5: Touches on a few relevant points but overall lacks clinical completeness.
- 1–3: Sparse coverage; misses most key staging or regimen considerations.
- 0: No comprehensiveness at all; completely superficial or irrelevant."""
    ),
    "knowledgeability": (
        "knowledgeability",
        "whether the response demonstrates rich and accurate oncology knowledge, including familiarity with AJCC staging criteria, NCCN/CSCO guidelines, landmark clinical trials, and lung cancer molecular subtypes",
        """Scoring Guide (0–10):
- 10: Demonstrates exceptional oncology knowledge with accurate staging and guideline alignment.
- 8–9: Shows clear domain knowledge with good guideline awareness; mostly accurate and relevant.
- 6–7: Displays some understanding of lung cancer staging and treatment but lacks depth or has notable gaps.
- 4–5: Limited oncology knowledge; understanding is basic or partially misaligned with guidelines.
- 1–3: Poor grasp of relevant clinical knowledge; superficial or mostly incorrect.
- 0: No evidence of meaningful oncology knowledge."""
    ),
    "correctness": (
        "correctness",
        "whether the TNM staging inference and recommended treatment regimens are clinically correct, guideline-compliant, and appropriate for the patient's histology and molecular profile",
        """Scoring Guide (0–10):
- 10: Fully correct and guideline-compliant; no errors in staging inference or staging-treatment alignment.
- 8–9: Mostly correct with minor deviations from standard staging criteria or clinical protocols.
- 6–7: Partially correct; some key staging errors or staging-treatment mismatches present.
- 4–5: Noticeable incorrect staging conclusions or regimen selections throughout.
- 1–3: Largely incorrect, clinically unsafe, or contradictory to standard oncology practice.
- 0: Entirely wrong or potentially harmful."""
    ),
    "relevance": (
        "relevance",
        "whether the staging inference and treatment recommendation are directly relevant to the patient's specific clinical profile derived from uploaded multimodal materials, including imaging findings, driver gene mutations, PS score, and PD-L1 expression",
        """Scoring Guide (0–10):
- 10: Fully tailored to the patient's clinical profile; highly relevant and actionable.
- 8–9: Mostly relevant; minor digressions but overall clinically useful.
- 6–7: Generally relevant, but includes generic or less patient-specific content.
- 4–5: Limited relevance; much of the response is non-specific or unhelpful.
- 1–3: Barely related to the patient's clinical situation or largely unhelpful.
- 0: Entirely irrelevant to the clinical context."""
    ),
    "diversity": (
        "diversity",
        "whether the response considers varied treatment options and alternative strategies across different staging scenarios, reflecting individualized and multi-scenario clinical thinking",
        """Scoring Guide (0–10):
- 10: Exceptionally rich; presents multiple treatment lines with well-reasoned alternative strategies across staging scenarios.
- 8–9: Considers a few alternative regimens or scenario-specific staging adjustments.
- 6–7: Some variety in options, but generally defaults to standard single-line recommendations.
- 4–5: Mostly formulaic; minimal consideration of alternative staging outcomes or treatment paths.
- 1–3: Very uniform or monotonous; no individualized clinical thinking.
- 0: No diversity or clinical individualization at all."""
    ),
    "logical_coherence": (
        "logical_coherence",
        "whether the clinical reasoning follows a coherent and medically valid sequential logic from multimodal data interpretation to TNM staging inference and then to treatment selection, without internal contradictions",
        """Scoring Guide (0–10):
- 10: Highly coherent; end-to-end reasoning from data to staging to treatment is clear, sequential, and contradiction-free.
- 8–9: Well-structured clinical logic with minor lapses in reasoning flow.
- 6–7: Some logical structure, but a few contradictory or weakly connected clinical inferences.
- 4–5: Often disorganized; staging and treatment reasoning are inconsistently linked.
- 1–3: Poorly structured; clinical logic is difficult to follow or internally contradictory.
- 0: Entirely illogical or clinically incoherent."""
    ),
    "factuality": (
        "factuality",
        "whether the cited staging criteria, drug names, clinical trial results, guideline recommendations, and molecular targets are factually accurate and up-to-date",
        """Scoring Guide (0–10):
- 10: All staging criteria, drug names, and guideline references are accurate and verifiable.
- 8–9: Mostly accurate; only minor factual issues in staging rules or trial citations.
- 6–7: Contains some factual inaccuracies in staging criteria or guideline references.
- 4–5: Several significant factual errors in staging conclusions or drug regimens.
- 1–3: Mostly fabricated or outdated clinical information.
- 0: Completely factually wrong or clinically dangerous throughout."""
    ),
}

        title, goal, rubric = descriptions[metric]

        return f"""---Role---

You are a helpful assistant evaluating the **{title}** of a generated response.

---Question---

{question}

---Golden Answers---

{str(answers)}

---Evaluation Goal---

Evaluate **{goal}** using a **0–10 integer scale**.

{rubric}

Output format:
<score>
your_score_here (an integer from 0 to 10)
</score>
<explanation>
Explain why you gave this score.
</explanation>

---Generation to be Evaluated---

{generation}
"""


    def score_extraction(metric, f1_score):
        try:
            prompt = build_prompt(metric)
            client, model_name = _get_client_and_model()

            payload = client.chat_completions(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    **{
                        k: v
                        for k, v in (globals().get("_completion_params", {}) or {}).items()
                        if v is not None
                    },
                },
                alias_retry=True,
                thinking_fallback=False,
            )
            choices = payload.get("choices", [])
            if not choices:
                raise RuntimeError(f"LLM响应缺少choices: {payload}")
            message = choices[0].get("message", {})
            content = extract_message_content(message.get("content", ""))
            
            score_str = content.split("<score>")[1].split("</score>")[0].strip()
            explanation = content.split("<explanation>")[1].split("</explanation>")[0].strip()
            score = int(score_str)
        except Exception as e:
            score = 5
            explanation = f"Failed to parse GPT output. Defaulted to score=5. Error: {str(e)}"

        score = score / 10
        score = (score + f1_score) / 2
        # 2 * score * f1_score / (score + f1_score) if (score + f1_score) > 0.0 else 0.0

        return metric, {"score": score, "explanation": explanation}

    metrics = [
        "comprehensiveness", "knowledgeability", "correctness",
        "relevance", "diversity", "logical_coherence", "factuality"
    ]

    # 使用 functools.partial 绑定 f1_score
    score_fn = partial(score_extraction, f1_score=f1_score)

    with ThreadPoolExecutor(max_workers=7) as executor:
        results = executor.map(score_fn, metrics)

    for metric, result in results:
        exp[metric] = result

    overall_score = round(np.mean([exp[m]["score"] for m in metrics]), 4)
    return {"score": overall_score, "explanation": exp}

