"""CLIP/SigLIP dual-tower encoder for the capacity ladder's weak rung — same API as MMRagEncoder
so train_rl / train_sft / eval_retrieval / mine_hard_negs work unchanged.

Query (image + question) = L2-normalized mean of the two tower embeddings (CLIP has no native
fusion); documents go through the text tower (hard 77/64-token cap — truncation is intrinsic to
this rung). LoRA adapters are attached over BOTH towers via one peft wrap of the full model, so
save_adapters / checkpoint_path behave exactly like the VLM path.
"""

import torch
import torch.nn.functional as F

_CLIP_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class CLIPStyleEncoder:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str = None,       # trained LoRA adapter dir (ours)
        device: str = "cuda:0",
        new_lora_r: int = 0,
        new_lora_alpha: int = 64,
        fusion: str = "mean",              # "mean" (img+text) | "image" (EchoSight-style)
        **_ignored,                        # max_len / query_instruction / style from the VLM path
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.fusion = fusion
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_name)
        tok = self.processor.tokenizer
        # SigLIP tokenizers report model_max_length=1e30 but the text tower has hard positional
        # capacity (64) — cap by the config, and pad to max_length (SigLIP's training convention).
        cfg_max = getattr(getattr(self.model.config, "text_config", None),
                          "max_position_embeddings", None) or 512
        self.text_max_len = min(getattr(tok, "model_max_length", 77) or 77, cfg_max, 512)
        self.pad_mode = "max_length" if "siglip" in self.model.config.model_type else True

        if checkpoint_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, checkpoint_path, is_trainable=False)
            if new_lora_r <= 0:
                self.model = self.model.merge_and_unload()
        if new_lora_r > 0:
            from peft import LoraConfig, get_peft_model

            if hasattr(self.model, "merge_and_unload"):
                self.model = self.model.merge_and_unload()
            lconf = LoraConfig(r=new_lora_r, lora_alpha=new_lora_alpha, lora_dropout=0.0,
                               target_modules=_CLIP_LORA_TARGETS, bias="none")
            self.model = get_peft_model(self.model, lconf)

        self.model = self.model.to(device).eval()

    # ---------------- encoding ----------------
    def _text_emb(self, texts):
        tok = self.processor.tokenizer(
            texts, padding=self.pad_mode, truncation=True, max_length=self.text_max_len,
            return_tensors="pt"
        ).to(self.device)
        emb = self.model.get_text_features(**tok)
        return F.normalize(emb.float(), dim=-1)

    def _image_emb(self, images):
        px = self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
        emb = self.model.get_image_features(pixel_values=px.to(self.device, dtype=torch.bfloat16))
        return F.normalize(emb.float(), dim=-1)

    def encode_queries(self, questions, images, batch_size=32, grad=False, to_cpu=False):
        out = []
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            for i in range(0, len(questions), batch_size):
                imgs = images[i : i + batch_size]
                ie = self._image_emb(imgs)
                if self.fusion == "image":
                    e = ie
                else:
                    te = self._text_emb(questions[i : i + batch_size])
                    e = F.normalize(ie + te, dim=-1)
                out.append(e.cpu() if to_cpu else e)
        return torch.cat(out, dim=0)

    def encode_docs(self, docs, batch_size=64, grad=False, to_cpu=False):
        out = []
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            for i in range(0, len(docs), batch_size):
                e = self._text_emb(list(docs[i : i + batch_size]))
                out.append(e.cpu() if to_cpu else e)
        return torch.cat(out, dim=0)

    # ---------------- training helpers (same surface as MMRagEncoder) ----------------
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def gradient_checkpointing_enable(self):
        # No-op: the towers are small, and reentrant GC with frozen (pixel) inputs detaches the
        # graph ("element 0 of tensors does not require grad") under LoRA-only training.
        pass

    def save_adapters(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

    @property
    def hidden_size(self):
        cfg = self.model.config
        return getattr(cfg, "projection_dim", None) or cfg.text_config.hidden_size

    @property
    def model_dot_modules(self):
        return self.model.modules
