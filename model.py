"""
HARA model training and inference module.
"""

from __future__ import annotations

import torch
import os
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Any
import json
from tqdm import tqdm

from config import Config
from data_loader import HARADataset, SYSTEM_PROMPT, get_collator


class TrainingLogCallback(TrainerCallback):
    """Training log callback: writes metrics such as loss/lr/eval to a file in real time."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        # Write the header
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("step,epoch,loss,eval_loss,learning_rate,grad_norm\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = round(state.epoch or 0.0, 2)
        loss = logs.get('loss', '')
        eval_loss = logs.get('eval_loss', '')
        lr = logs.get('learning_rate', '')
        grad_norm = logs.get('grad_norm', '')
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"{step},{epoch},{loss},{eval_loss},{lr},{grad_norm}\n")


class BestModelCallback(TrainerCallback):
    """Automatically save the best checkpoint to the best/ directory after each evaluation."""

    def __init__(self, best_dir: str, save_fn):
        self.best_dir = best_dir
        self.save_fn = save_fn
        self.best_eval_loss = float("inf")
        self.best_step = 0

    def on_evaluate(self, args, state, control, **kwargs):
        """After evaluation, save the checkpoint if eval_loss is better."""
        logs = state.log_history[-1] if state.log_history else {}
        eval_loss = logs.get("eval_loss")
        if eval_loss is None:
            return
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = state.global_step
            self.save_fn(self.best_dir)
            print(f"  [BestModel] Step {state.global_step}: eval_loss={eval_loss:.6f} -> saved to {self.best_dir}")


class HARAModel:
    """HARA model class."""

    def __init__(self, config: Config):
        """
        Initialize the model.

        Args:
            config: configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Resolve the model path (supports ModelScope)
        model_path = self._resolve_model_path(config.MODEL_NAME)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=self._get_cache_dir()
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model lazily, when train()/load_model() is called
        self.model = None

    def _get_cache_dir(self) -> str | None:
        """Return the cache directory corresponding to MODEL_SOURCE."""
        if self.config.MODEL_SOURCE == "modelscope":
            return self.config.MODELSCOPE_CACHE_DIR
        else:
            return self.config.HUGGINGFACE_CACHE_DIR

    def _resolve_model_path(self, model_name: str) -> str:
        """
        Resolve the model path according to MODEL_SOURCE.

        For modelscope, download the model locally via snapshot_download and return the path;
        for huggingface, return model_name directly.
        """
        if self.config.MODEL_SOURCE == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError:
                raise ImportError(
                    "Please install modelscope: pip install modelscope"
                )
            cache_dir = self.config.MODELSCOPE_CACHE_DIR
            print(f"Downloading model from ModelScope: {model_name}")
            local_path = snapshot_download(model_name, cache_dir=cache_dir)
            print(f"ModelScope model path: {local_path}")
            return local_path
        else:
            return model_name

    def _apply_lora(self):
        """Apply LoRA fine-tuning."""
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self, train_dataset: HARADataset, val_dataset: HARADataset):
        """
        Train the model.

        Args:
            train_dataset: training dataset
            val_dataset: validation dataset
        """
        # Clear the GPU cache
        torch.cuda.empty_cache()
        
        # Load the model (8-bit quantization to reduce VRAM usage)
        print("Loading model (8-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model_path = self._resolve_model_path(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self._get_cache_dir()
        )
        
        # Enable gradient checkpointing to reduce memory usage
        self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        if self.config.USE_LORA:
            self._apply_lora()
        
        # Create the output directory (timestamped subdirectory for logs)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging_dir = os.path.join(self.config.OUTPUT_DIR, f"logs_{run_time}")
        os.makedirs(logging_dir, exist_ok=True)
        print(f"Training log directory: {logging_dir}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            logging_dir=logging_dir,
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none"
        )

        # Data collator (dynamic padding)
        data_collator = get_collator(self.tokenizer)

        # Create the log callback
        log_callback = TrainingLogCallback(os.path.join(logging_dir, "training_log.csv"))

        # Create the best-model saving callback
        best_dir = os.path.join(self.config.OUTPUT_DIR, "best")
        best_callback = BestModelCallback(best_dir, save_fn=lambda path: (
            self.model.save_pretrained(path),
            self.tokenizer.save_pretrained(path)
        ))

        # Create the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[log_callback, best_callback],
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save the final model
        trainer.save_model(os.path.join(self.config.OUTPUT_DIR, "final_model"))
        print(f"Model saved to {self.config.OUTPUT_DIR}/final_model")

    def load_model(self, model_path: str):
        """
        Load the trained fine-tuned model (LoRA).

        Args:
            model_path: path to the LoRA adapter
        """
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        base_model_path = self._resolve_model_path(self.config.MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self._get_cache_dir()
        )

        self.model = PeftModel.from_pretrained(base_model, model_path)
        print(f"Loaded fine-tuned model (LoRA): {model_path}")

    def load_base_model(self):
        """
        Load the pure base model (without any LoRA / fine-tuned weights).

        Used for zero-shot inference evaluation, performing HARA analysis directly with the pretrained model.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model_path = self._resolve_model_path(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self._get_cache_dir()
        )

        print(f"Loaded base model (inference only, no LoRA): {self.config.MODEL_NAME}")

    def predict(self, dataset: HARADataset, output_path: str = None) -> list[dict[str, Any]]:
        """
        Run batched inference on a dataset.

        Args:
            dataset: dataset
            output_path: output file path (saved incrementally in real time)

        Returns:
            list of prediction results
        """
        self.model.eval()
        predictions = []

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[\n')

        # Use left padding during inference
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        batch_size = self.config.INFERENCE_BATCH_SIZE
        print(f"Starting batched inference (batch_size={batch_size})...")

        total = len(dataset)
        for batch_start in tqdm(range(0, total, batch_size), desc="Inference progress"):
            batch_end = min(batch_start + batch_size, total)
            batch_items = [dataset[i] for i in range(batch_start, batch_end)]

            # Build batch prompts
            prompts = []
            for item in batch_items:
                user_content = self._build_user_message(item['input'])
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)

            # Encode the batch (left padding, uniform length within the batch)
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                max_length=self.config.MAX_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)

            # Batch generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    num_beams=self.config.NUM_BEAMS,
                    do_sample=True if self.config.TEMPERATURE > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode each sample's output in the batch (keep only the newly generated part)
            input_len = inputs['input_ids'].shape[-1]
            for idx, item in enumerate(batch_items):
                generated_ids = outputs[idx][input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                reasoning, prediction = self._split_reasoning_and_output(generated_text)

                result = {
                    'id': item['id'],
                    'input': item['input'],
                    'output': prediction,
                    'reasoning': reasoning
                }
                predictions.append(result)

                # Save to the file incrementally
                if output_path:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        global_idx = batch_start + idx
                        if global_idx < total - 1:
                            f.write(',\n')
                        else:
                            f.write('\n]')

        # Restore the tokenizer's original padding_side
        self.tokenizer.padding_side = original_padding_side
        print(f"Batched inference completed, processed {len(predictions)} samples")

        return predictions

    def _build_user_message(self, input_data: dict[str, str]) -> str:
        """Build the user message - present the input data in a structured form."""
        lines = ["请根据以下车辆失效场景信息，进行 HARA 危害分析与风险评估：\n"]
        for key, value in input_data.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("\n请按要求的分析步骤输出结果。")
        return "\n".join(lines)

    def _split_reasoning_and_output(self, generated_text: str) -> tuple:
        """
        Split the thinking reasoning and the JSON output.

        Args:
            generated_text: full text generated by the model

        Returns:
            (reasoning: str, prediction_dict: dict)
        """
        # Try to extract the <think> ... </think> section
        think_start = "<think>"
        think_end = "</think>"
        
        if think_start in generated_text and think_end in generated_text:
            start_idx = generated_text.find(think_start) + len(think_start)
            end_idx = generated_text.find(think_end)
            reasoning = generated_text[start_idx:end_idx].strip()
            
            # Extract the JSON after </think>
            after_thinking = generated_text[end_idx + len(think_end):].strip()
            prediction = self._parse_json_from_text(after_thinking)
        else:
            # No thinking markers found; treat the entire text as reasoning
            reasoning = generated_text
            prediction = self._parse_json_from_text(generated_text)
        
        return reasoning, prediction

    def _parse_json_from_text(self, text: str) -> dict[str, str]:
        """Parse a JSON object from text."""
        import json
        
        try:
            json_start = text.find('{')
            if json_start != -1:
                brace_count = 1
                json_end = json_start + 1
                while json_end < len(text) and brace_count > 0:
                    if text[json_end] == '{':
                        brace_count += 1
                    elif text[json_end] == '}':
                        brace_count -= 1
                    json_end += 1
                
                if brace_count == 0:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {}

    def _parse_prediction(self, generated_text: str, prompt: str) -> dict[str, str]:
        """
        Parse the generated prediction (kept for backward compatibility with the old interface).
        """
        import json
        
        # Extract from the generated text (strip the prompt portion)
        if prompt in generated_text:
            content = generated_text[len(prompt):].strip()
        else:
            content = generated_text.strip()
        
        # Try to parse the JSON
        try:
            # Find the first complete JSON object
            json_start = content.find('{')
            if json_start != -1:
                # Find the matching closing brace (handling nesting)
                brace_count = 1
                json_end = json_start + 1
                while json_end < len(content) and brace_count > 0:
                    if content[json_end] == '{':
                        brace_count += 1
                    elif content[json_end] == '}':
                        brace_count -= 1
                    json_end += 1
                
                if brace_count == 0:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    return parsed
        except json.JSONDecodeError:
            pass
        
        # Parsing failed; return an empty dict
        return {}

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: str):
        """
        Save the prediction results.

        Args:
            predictions: list of prediction results
            output_path: output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {output_path}")
