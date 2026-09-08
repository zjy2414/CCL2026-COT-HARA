"""
Data loading and preprocessing module - supports Chain-of-Thought (CoT) inference mode.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Any
from transformers import DataCollatorForSeq2Seq


# Load the system prompt from CoT.md
_COT_PATH = os.path.join(os.path.dirname(__file__), "CoT.md")
with open(_COT_PATH, "r", encoding="utf-8") as _f:
    _cot_md = _f.read()
SYSTEM_PROMPT = _cot_md


# ASIL lookup table (consistent with the unified table in CoT.md, ISO 26262 Part 3)
# Rows = (S, E) combinations, columns = C values
# Rule: S=0 / E=0 / C=0 -> QM
ASIL_LOOKUP = {
    (1, 1): {1: "QM", 2: "QM", 3: "QM"},
    (1, 2): {1: "QM", 2: "QM", 3: "QM"},
    (1, 3): {1: "QM", 2: "QM", 3: "A"},
    (1, 4): {1: "QM", 2: "A",  3: "B"},
    (2, 1): {1: "QM", 2: "QM", 3: "QM"},
    (2, 2): {1: "QM", 2: "QM", 3: "A"},
    (2, 3): {1: "QM", 2: "A",  3: "B"},
    (2, 4): {1: "A",  2: "B",  3: "C"},
    (3, 1): {1: "QM", 2: "QM", 3: "A"},
    (3, 2): {1: "QM", 2: "A",  3: "B"},
    (3, 3): {1: "A",  2: "B",  3: "C"},
    (3, 4): {1: "B",  2: "C",  3: "D"},
}

ASIL_ORDER = {"QM": 0, "A": 1, "B": 2, "C": 3, "D": 4}


def lookup_asil(s: int, e: int, c: int) -> str:
    """
    Look up the ASIL level, consistent with the unified lookup table in CoT.md.

    Args:
        s: Severity (0-3)
        e: Exposure (0-4)
        c: Controllability (0-3)

    Returns:
        ASIL level string: QM / A / B / C / D
    """
    # S=0 (no harm), E=0 (extremely low probability), or C=0 (negligible risk) -> QM
    if s == 0 or e == 0 or c == 0:
        return "QM"
    # (S, E) combinations not listed in the table default to QM
    return ASIL_LOOKUP.get((s, e), {}).get(c, "QM")


class HARADataset(Dataset):
    """HARA dataset class - supports the CoT chain-of-thought format."""

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 4096):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, data_path: str) -> list[dict[str, Any]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]

        user_content = self._build_user_message(item['input'])
        
        # Build the assistant reply: reasoning + JSON output
        reasoning = self._build_reasoning(item['input'], item['output'])
        target_json = json.dumps(item['output'], ensure_ascii=False, indent=2)
        assistant_content = f"<think>\n{reasoning}\n</think>\n\n```json\n{target_json}\n```"

        if self.tokenizer:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,  # dynamic padding is handled at batch level by the DataCollator
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            labels = input_ids.clone()

            # Compute loss only on the assistant reply portion
            assistant_start_token_ids = self.tokenizer.encode("<|im_start|>\nassistant\n", add_special_tokens=False)
            start_pos = self._find_subsequence(input_ids, assistant_start_token_ids)
            if start_pos != -1:
                labels[:start_pos] = -100
            else:
                alt_pattern = self.tokenizer.encode("assistant", add_special_tokens=False)
                pos = self._find_subsequence(input_ids, alt_pattern)
                if pos != -1:
                    labels[:pos] = -100

            pad_token_id = self.tokenizer.pad_token_id
            labels[attention_mask == 0] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'id': item['id'],
                'prompt': self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True
                )
            }
        else:
            return {
                'id': item['id'],
                'input': item['input'],
                'output': item['output'],
                'prompt': user_content
            }

    def _find_subsequence(self, tensor, subseq) -> int:
        """Find the start position of a subsequence within a tensor."""
        n, m = len(tensor), len(subseq)
        if m > n:
            return -1
        for i in range(n - m + 1):
            if all(tensor[i:i+m][j] == subseq[j] for j in range(m)):
                return i + m
        return -1

    def _build_user_message(self, input_data: dict[str, str]) -> str:
        """Build the user message - present the input data in a structured form."""
        lines = ["请根据以下车辆失效场景信息，进行 HARA 危害分析与风险评估：\n"]
        for key, value in input_data.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("\n请按要求的分析步骤输出结果。")
        return "\n".join(lines)

    def _build_reasoning(self, inp: dict[str, str], out: dict[str, str]) -> str:
        """Build a complete CoT reasoning trace from the input/output data (used as a supervision signal during training)."""
        e_val = out.get("Exposure or Frequency 'E'", "?")
        s_val = out.get("Severity 'S'", "?")
        c_val = out.get("Control ability 'C'", "?")
        asil = out.get("Resulting A SIL", "")

        driver_status = "在" if inp.get("Driver in car or not") == "是" else "不在"
        dist_near = "小于" in inp.get("relative position", "")

        # Build the explanatory rationale for C
        c_parts = [
            f"驾驶员{'在车内可及时反应' if driver_status == '在' else '不在车内无法主动控制'}",
            ("距离近反应时间受限" if dist_near else "距离远有充足反应时间"),
        ]
        try:
            ego_spd = int(inp.get("ego vehicle speed", "0kph").replace("kph", "").strip())
            if ego_spd > 60:
                c_parts.append(f"高速行驶(>{ego_spd}kph)增加控制难度")
        except (ValueError, TypeError):
            pass
        if "弱势" in inp.get("Nearby elements", ""):
            c_parts.append("涉及弱势交通参与者需格外谨慎")

        # Validate ASIL consistency (the lookup result should match the ground truth)
        try:
            s_int = int(s_val)
            e_int = int(e_val)
            c_int = int(c_val)
            expected_asil = lookup_asil(s_int, e_int, c_int)
            if expected_asil != asil:
                # If the label disagrees with the lookup table, trust the label (make no change)
                expected_asil = asil
        except (ValueError, TypeError):
            expected_asil = asil

        return (
            "**Step 1: 场景理解**\n"
            f"当前场景为{inp.get('Road Layout', '')}上的{inp.get('Malfunction', '')}失效事件。"
            f"车辆处于{inp.get('Manoeuvre', '')}状态，自车速度{inp.get('ego vehicle speed', '')}，"
            f"周边存在{inp.get('Nearby elements', '')}，相对位置为{inp.get('relative position', '')}。"
            f"气象条件：{inp.get('Weather', '')}，能见度：{inp.get('Visibility', '')}"
            f"，驾驶员{driver_status}车内。\n\n"

            "**Step 2: 危害事件推导**\n"
            f"由于{inp.get('vehicle hazard', '')}，结合道路环境和运行状态，将导致：{out.get('hazardous event', '')}\n"
            f"可能的直接危害为：{out.get('Possible Hazard', '')}\n\n"

            "**Step 3: 风险等级评定**\n"
            f"- Exposure (E)={e_val}: 该场景的出现频率对应等级 {e_val}\n"
            f"- Severity (S)={s_val}: 潜在伤害严重程度对应等级 {s_val}\n"
            f"- Controllability (C)={c_val}: {'；'.join(c_parts)} → 综合判定C={c_val}\n\n"

            "**Step 4: ASIL 判定（查表）**\n"
            f"由 ISO 26262 ASIL 查找表：S={s_val}, E={e_val}, C={c_val} → ASIL = {expected_asil}\n"
            + (f"ASIL 等级为 QM（质量管理），无需定义 Safety Goal 和 FTTI\n" if expected_asil == "QM"
               else f"ASIL 等级为 {expected_asil}，需要设定安全目标\nSafety Goal: {out.get('Safety Goal', '')}, FTTI: {out.get('FTTI', '')}\n")
        )


def load_datasets(config, tokenizer=None):
    """Load the training set and split out a validation set from it (val.json is A-leaderboard data with empty outputs)."""
    # Only load train.json
    full_train = HARADataset(config.TRAIN_DATA_PATH, tokenizer=tokenizer, max_length=config.MAX_LENGTH)

    # Split into train and validation sets by the configured ratio
    ratio = config.TRAIN_SPLIT_RATIO
    split_idx = int(len(full_train.data) * ratio)
    train_data = full_train.data[:split_idx]
    val_data = full_train.data[split_idx:]

    train_dataset = HARADataset.__new__(HARADataset)
    train_dataset.data = train_data
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = full_train.max_length

    val_dataset = HARADataset.__new__(HARADataset)
    val_dataset.data = val_data
    val_dataset.tokenizer = tokenizer
    val_dataset.max_length = full_train.max_length

    print(f"Training samples: {len(train_dataset)} ({ratio:.0%})")
    print(f"Validation samples: {len(val_dataset)} ({(1 - ratio):.0%})")

    return train_dataset, val_dataset


def load_test_dataset(data_path):
    """Load the test set."""
    test_dataset = HARADataset(data_path)
    print(f"Test samples: {len(test_dataset)}")
    return test_dataset


class HARACollator(DataCollatorForSeq2Seq):
    """DataCollator with dynamic padding, ensuring pad positions in labels are aligned to -100."""

    def __call__(self, features):
        # Pop out non-tensor fields (prompt, id) to stay compatible with Trainer internals
        ids = [f.pop('id', None) for f in features]
        prompts = [f.pop('prompt', None) for f in features]

        batch = super().__call__(features)
        if any(i is not None for i in ids):
            batch['id'] = ids
        if any(p is not None for p in prompts):
            batch['prompt'] = prompts
        return batch


def get_collator(tokenizer):
    """Get the HARA-specific dynamic-padding DataCollator."""
    return HARACollator(tokenizer=tokenizer, padding=True, max_length=None, return_tensors='pt')
