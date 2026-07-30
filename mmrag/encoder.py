"""Thin encode API over VLM2Vec's MMEBModel for the mm-RAG project.

Wraps model loading (base / +VLM2Vec LoRA checkpoint / +fresh trainable LoRA for RL/SFT) and
(image+text | text-only) batch encoding through the repo's own processor path, so query/passage
embeddings match VLM2Vec's training conventions (image token + instruction inside the text,
'last' pooling, L2-normalized).

Used by eval_retrieval.py (frozen), train_sft.py and train_rl.py (trainable LoRA adapters).
"""

import torch

from src.arguments import ModelArguments
from src.model.model import MMEBModel
from src.model.processor import (
    QWEN2_VL, VLM_IMAGE_TOKENS, get_backbone_name, load_processor, process_vlm_inputs_fns,
)

# Default instructions (VLM2Vec/MMEB style: instruction lives inside the query text; targets are
# plain passages). The image token is prepended automatically in encode_queries.
QUERY_INSTRUCTION = "Represent the given image with the following question: {q}"
DOC_INSTRUCTION = "{d}"

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MMRagEncoder:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str = None,     # VLM2Vec LoRA adapter dir (merged in unless trainable)
        device: str = "cuda:0",
        pooling: str = "last",
        normalize: bool = True,
        max_len: int = 2048,
        new_lora_r: int = 0,             # >0: attach FRESH trainable LoRA adapters (RL / SFT arm)
        new_lora_alpha: int = 64,
        query_instruction: str = QUERY_INSTRUCTION,
    ):
        self.device = device
        self.query_instruction = query_instruction
        self.max_len = max_len

        margs = ModelArguments(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            lora=bool(checkpoint_path),
            pooling=pooling,
            normalize=normalize,
        )
        # is_trainable=False merges any VLM2Vec adapter into the base -> a clean frozen encoder.
        self.model = MMEBModel.load(margs, is_trainable=False)
        self.backbone = margs.model_backbone
        assert self.backbone is not None
        self.process_fn = process_vlm_inputs_fns[self.backbone]
        self.image_token = VLM_IMAGE_TOKENS[self.backbone]

        margs_p = ModelArguments(model_name=model_name, pooling=pooling, normalize=normalize)
        margs_p.model_backbone = self.backbone

        class _DA:  # load_processor only reads these fields
            max_len = self.max_len
            resize_use_processor = True
            resize_min_pixels = 28 * 28 * 4
            resize_max_pixels = 28 * 28 * 1280

        self.processor = load_processor(margs_p, _DA())

        if new_lora_r > 0:
            from peft import LoraConfig, get_peft_model

            lconf = LoraConfig(r=new_lora_r, lora_alpha=new_lora_alpha, lora_dropout=0.0,
                               target_modules=_LORA_TARGETS, bias="none")
            self.model.encoder = get_peft_model(self.model.encoder, lconf)
            self.model.encoder.enable_input_require_grads()

        self.model = self.model.to(device, dtype=torch.bfloat16)
        self.model.eval()

    # ---------------- forward path ----------------
    def _forward(self, texts, images):
        """One processor+encoder pass. images: list of PIL.Image or None (len == len(texts))."""
        inputs = self.process_fn({"text": texts, "images": images}, processor=self.processor,
                                 max_length=self.max_len)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        # MMEBModel.encode_input for qwen2_vl goes through the generic HF branch: strip helper keys
        for k in ("texts", "images"):
            inputs.pop(k, None)
        # collate per-sample visual tensors like the eval collator does
        inputs = _merge_visual(inputs, self.device)
        return self.model.encode_input(inputs)  # (B, d), normalized if normalize=True

    def encode(self, texts, images=None, batch_size=16, grad=False, to_cpu=False):
        """Encode raw (text, image) pairs. Returns (N, d) tensor (fp32 on CPU if to_cpu)."""
        images = images if images is not None else [None] * len(texts)
        out = []
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            for i in range(0, len(texts), batch_size):
                e = self._forward(texts[i : i + batch_size], images[i : i + batch_size])
                out.append(e.float().cpu() if to_cpu else e)
        return torch.cat(out, dim=0)

    def format_query(self, question):
        return f"{self.image_token} {self.query_instruction.format(q=question)}"

    def encode_queries(self, questions, images, batch_size=8, grad=False, to_cpu=False):
        texts = [self.format_query(q) for q in questions]
        return self.encode(texts, images, batch_size=batch_size, grad=grad, to_cpu=to_cpu)

    def encode_docs(self, docs, batch_size=32, grad=False, to_cpu=False):
        return self.encode(list(docs), None, batch_size=batch_size, grad=grad, to_cpu=to_cpu)

    # ---------------- training helpers ----------------
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def gradient_checkpointing_enable(self):
        self.model.encoder.gradient_checkpointing_enable()
        if hasattr(self.model.encoder, "enable_input_require_grads"):
            self.model.encoder.enable_input_require_grads()

    def save_adapters(self, output_dir):
        self.model.encoder.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

    @property
    def hidden_size(self):
        return self.model.rep_dim


def _merge_visual(inputs, device):
    """Qwen2_VL_process_fn returns per-sample lists for pixel_values/image_grid_thw; the model
    expects them concatenated (or absent for text-only batches). Mirrors EvalCollator's usage."""
    import numpy as np

    for pk, gk in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
        vals = inputs.get(pk)
        if vals is None:
            continue
        keep = [v for v in vals if v is not None]
        if not keep:
            inputs.pop(pk, None)
            inputs.pop(gk, None)
            continue
        pv = np.concatenate([v for v in keep], axis=0)
        gt = np.concatenate([g for g in inputs[gk] if g is not None], axis=0)
        inputs[pk] = torch.from_numpy(pv).to(device)
        inputs[gk] = torch.from_numpy(gt).to(device)
    return inputs
