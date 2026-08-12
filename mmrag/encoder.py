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

# GME (Alibaba-NLP/gme-Qwen2-VL-*) embeds everything through a fixed chat wrapper and pools the
# final <|endoftext|> token (see src/model/baseline_backbone/gme/gme_inference.py::embed). Image
# comes FIRST inside the user turn; documents use the default instruction.
GME_TEMPLATE = ("<|im_start|>system\n{instr}<|im_end|>\n"
                "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>")
GME_DEFAULT_INSTRUCTION = "You are a helpful assistant."
GME_QUERY_INSTRUCTION = "Find a Wikipedia paragraph that answers the question about the image."
GME_IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

# Named encoder profiles: (base model, adapter checkpoint, text-format style).
# 7B bases live in plain local dirs (login-node killer OOM-kills python hub downloads; curl-fetched).
_MODELS_DIR = "/lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/models"
ENCODER_PROFILES = {
    "gme2b": {"model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "checkpoint": None, "style": "gme"},
    "gme7b": {"model": f"{_MODELS_DIR}/gme-Qwen2-VL-7B-Instruct", "checkpoint": None, "style": "gme"},
    "vlm2vec2b": {"model": "Qwen/Qwen2-VL-2B-Instruct",
                  "checkpoint": "TIGER-Lab/VLM2Vec-Qwen2VL-2B", "style": "vlm2vec"},
    "vlm2vec7b": {"model": f"{_MODELS_DIR}/Qwen2-VL-7B-Instruct",
                  "checkpoint": "TIGER-Lab/VLM2Vec-Qwen2VL-7B", "style": "vlm2vec"},
    # capacity-ladder weak rungs: dual-tower encoders (mmrag/clip_encoder.py)
    "clip": {"model": "openai/clip-vit-large-patch14-336", "checkpoint": None, "style": "clip"},
    "siglip2": {"model": "google/siglip2-so400m-patch16-384", "checkpoint": None, "style": "clip"},
}


def load_encoder(profile: str, checkpoint_path="__profile__", **kwargs):
    """Profile-dispatching factory: VLM profiles -> MMRagEncoder, clip-style -> CLIPStyleEncoder.
    Extra kwargs irrelevant to a family are ignored by that family's constructor."""
    p = ENCODER_PROFILES[profile]
    ckpt = p["checkpoint"] if checkpoint_path == "__profile__" else checkpoint_path
    if p["style"] == "clip":
        from mmrag.clip_encoder import CLIPStyleEncoder

        return CLIPStyleEncoder(p["model"], checkpoint_path=ckpt, **kwargs)
    return MMRagEncoder(p["model"], checkpoint_path=ckpt, style=p["style"], **kwargs)

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MMRagEncoder:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str = None,     # VLM2Vec LoRA adapter dir (merged in unless trainable)
        device: str = "cuda:0",
        pooling: str = "last",
        normalize: bool = True,
        max_len: int = 512,              # doc-side (text-only) token budget
        query_max_len: int = 2048,       # query-side budget: image pads (<=1280) + text; truncating
                                         # image pads breaks the vision-feature count check
        new_lora_r: int = 0,             # >0: attach FRESH trainable LoRA adapters (RL / SFT arm)
        new_lora_alpha: int = 64,
        query_instruction: str = None,
        style: str = "vlm2vec",          # text-format convention: "vlm2vec" | "gme"
    ):
        self.device = device
        self.style = style
        self.query_instruction = query_instruction or (
            GME_QUERY_INSTRUCTION if style == "gme" else QUERY_INSTRUCTION)
        self.max_len = max_len
        self.query_max_len = query_max_len

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
    def _forward(self, texts, images, max_length=None):
        """One processor+encoder pass. images: list of PIL.Image or None (len == len(texts))."""
        inputs = self.process_fn({"text": texts, "images": images}, processor=self.processor,
                                 max_length=max_length or self.max_len)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        # MMEBModel.encode_input for qwen2_vl goes through the generic HF branch: strip helper keys
        for k in ("texts", "images"):
            inputs.pop(k, None)
        # collate per-sample visual tensors like the eval collator does
        inputs = _merge_visual(inputs, self.device)
        # Compute mrope position_ids OURSELVES with a properly-shaped (n_images, 3) grid tensor.
        # The vendored forward's own get_rope_index call indexes the per-sample grid LIST as if it
        # were that tensor and crashes — but only when rope_deltas is still uncached (first image
        # batch of the process), which is why runs that encode the text corpus first "worked"
        # (with approximate cached-delta positions). Passing position_ids skips that branch.
        if inputs.get("image_grid_thw") is not None:
            import numpy as np

            grids = [g for g in inputs["image_grid_thw"] if g is not None]
            if grids:
                gt = torch.cat([torch.from_numpy(g) if isinstance(g, np.ndarray) else g
                                for g in grids]).to(self.device)
                base = self.model.encoder  # PeftModel delegates get_rope_index to the base model
                pos, _ = base.get_rope_index(inputs["input_ids"], gt, None,
                                             inputs.get("attention_mask"))
                inputs["position_ids"] = pos
        return self.model.encode_input(inputs)  # (B, d), normalized if normalize=True

    def encode(self, texts, images=None, batch_size=16, grad=False, to_cpu=False, max_length=None):
        """Encode raw (text, image) pairs. Returns (N, d) tensor (fp32 on CPU if to_cpu)."""
        images = images if images is not None else [None] * len(texts)
        out = []
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            for i in range(0, len(texts), batch_size):
                e = self._forward(texts[i : i + batch_size], images[i : i + batch_size], max_length)
                out.append(e.float().cpu() if to_cpu else e)
        return torch.cat(out, dim=0)

    def format_query(self, question):
        if self.style == "gme":
            return GME_TEMPLATE.format(instr=self.query_instruction,
                                       content=GME_IMAGE_TOKENS + question)
        return f"{self.image_token} {self.query_instruction.format(q=question)}"

    def format_doc(self, doc):
        if self.style == "gme":
            return GME_TEMPLATE.format(instr=GME_DEFAULT_INSTRUCTION, content=doc)
        return doc

    def encode_queries(self, questions, images, batch_size=8, grad=False, to_cpu=False):
        texts = [self.format_query(q) for q in questions]
        return self.encode(texts, images, batch_size=batch_size, grad=grad, to_cpu=to_cpu,
                           max_length=self.query_max_len)

    def encode_docs(self, docs, batch_size=32, grad=False, to_cpu=False):
        return self.encode([self.format_doc(d) for d in docs], None,
                           batch_size=batch_size, grad=grad, to_cpu=to_cpu)

    @classmethod
    def from_profile(cls, profile: str, checkpoint_path="__profile__", **kwargs):
        """Build from a named profile ('gme2b', 'vlm2vec2b', ...). Pass checkpoint_path explicitly
        (e.g. a trained adapter dir) to override the profile's default adapter."""
        p = ENCODER_PROFILES[profile]
        ckpt = p["checkpoint"] if checkpoint_path == "__profile__" else checkpoint_path
        return cls(p["model"], checkpoint_path=ckpt, style=p["style"], **kwargs)

    def ref_ctx(self):
        """Context manager that disables the trainable LoRA adapters, exposing the reference
        (initial) policy — valid because LoRA-B is zero-initialised, so init policy == base."""
        import contextlib

        m = self.model.encoder
        return m.disable_adapter() if hasattr(m, "disable_adapter") else contextlib.nullcontext()

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
    """Qwen2_VL_process_fn returns per-sample LISTS for pixel_values/image_grid_thw (None for
    text-only samples). The vendored Qwen2VL forward indexes these lists per sample and
    concatenates internally — so keep them as lists; only drop the keys for all-text batches
    (the model would try to iterate a list of Nones otherwise)."""
    for pk, gk in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
        vals = inputs.get(pk)
        if vals is None or all(v is None for v in vals):
            inputs.pop(pk, None)
            inputs.pop(gk, None)
    return inputs
