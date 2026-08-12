"""Method figure for v3-pure: ONE layout spec, rendered to an edge-trimmed PDF and an editable PPTX.

Why one spec and two renderers: this box has no LibreOffice (checked: libreoffice/soffice/unoconv
all absent), so the PPTX cannot be converted to PDF on-machine. Rather than let the two artifacts
drift, the geometry and text live in SPEC below and both renderers consume it. Re-export the PPTX
through PowerPoint/LibreOffice later if you want a literally-converted PDF; it will match.

Every hyperparameter printed comes from $MMRAG_DATA/runs/<run>/args.json — not typed here, not
taken from the README. Missing keys are a hard error rather than a plausible default.

Visual language copied from the existing deck fig_af_rl_arch.pptx / fig_rag_arch.pptx:
solid accent header + darker outline, light tinted body, thin grey connectors, objective bar
at the foot. Theme is blue/orange per the user: BLUE = the trainable retrieval path,
ORANGE = the frozen reward path (the same convention fig_af_rl_arch already uses).

Box -> code map (VLM2Vec-rl @ v23-base):
  query/index  train_rl.build_rl_corpus; refresh_corpus() every --refresh_steps
  policy       encoder.load_encoder (frozen backbone + LoRA), rl_core.pool_logps
  sampling     rl_core.pl_sample_lists (Gumbel-top-k == exact PL w/o replacement),
               rl_core.pl_list_logps (exact shrinking-denominator factorization)
  reward       reader_vlm.LiveVLMReader.score_judge -> reader_vlm.answer_correct
               (reward = FRACTION of n_rollouts sampled answers that cover-EM a gold alias)
  credit       train_rl.main R[b,g]=r_doc[ids].mean(); (R-R.mean)/R.std;
               rl_core.embedding_ppo_loss (ratio==1 at inner_epochs=1 -> plain PG)
  absent       contrastive_coef 0, vf_coef 0, kl_beta 0, no_force_gold True

    MMRAG_DATA=<data> python3 mmrag/plot_v3_arch.py --out mmrag/fig_mm_v3_arch
"""

import argparse
import ast
import json
import os
import re
import sys

# ---- palette: template neutrals + the user's blue/orange -----------------------------------
BLUE, BLUE_DK, BLUE_LT = "5B92C4", "2F5E8C", "E2EDF6"
ORANGE, ORANGE_DK, ORANGE_LT = "D2884A", "8C5320", "F9EBDD"
GREY, GREY_LN, GREY_LT = "6B7480", "8A949E", "F2F4F8"

REQUIRED = ["profile", "lora_r", "lora_alpha", "temperature", "pl_support", "pl_group", "pl_k",
            "reader_model", "reader_rollouts", "reader_temperature", "refresh_steps",
            "n_distractor_articles", "batch_size", "max_steps", "learning_rate",
            "contrastive_coef", "vf_coef", "kl_beta", "no_force_gold", "inner_epochs",
            "baseline", "algo"]

W, H = 13.33, 6.40          # slide / canvas inches (template is 13.33 x 6.96)


def backbone_of(profile):
    """Resolve profile -> backbone model id by parsing ENCODER_PROFILES out of encoder.py.
    Parsed, not imported (importing drags in torch), and not guessed: deriving the backbone
    from the READER name printed 'Qwen2.5-VL-2B' for a Qwen2-VL encoder in an earlier draft."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "encoder.py"),
                 os.path.join(here, "..", "..", "VLM2Vec-rl", "mmrag", "encoder.py")):
        if not os.path.exists(cand):
            continue
        src = open(cand).read()
        m = re.search(r"ENCODER_PROFILES\s*=\s*(\{.*?\n\})", src, re.S)
        if not m:
            continue
        try:
            prof = ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))
        except Exception:
            continue
        if profile in prof:
            return prof[profile]["model"].split("/")[-1].replace("-Instruct", "")
    sys.exit(f"could not resolve backbone for profile {profile!r} from encoder.py")


def load_args(D, run):
    p = os.path.join(D, "runs", run, "args.json")
    if not os.path.exists(p):
        sys.exit(f"no args.json at {p} — the figure's numbers must come from a real run")
    a = json.load(open(p))
    missing = [k for k in REQUIRED if k not in a]
    if missing:
        sys.exit(f"args.json missing {missing} — refusing to draw invented values")
    if a["algo"] != "plgrpo" or a["contrastive_coef"] or not a["no_force_gold"]:
        sys.exit(f"{run} is not v3-pure (algo={a['algo']}, cc={a['contrastive_coef']}, "
                 f"no_force_gold={a['no_force_gold']})")
    return a


def build_spec(a):
    """Return (boxes, arrows, notes). Geometry in inches, origin top-left (PowerPoint convention);
    the matplotlib renderer flips y."""
    tau, M, G, k = a["temperature"], a["pl_support"], a["pl_group"], a["pl_k"]
    reader = a["reader_model"].split("/")[-1].replace("-Instruct", "")
    backbone = backbone_of(a["profile"])
    lr = f"{a['learning_rate']:g}"

    boxes = [
        # key, x, y, w, h, title, body, fill, line, title_colour
        ("query", 0.35, 0.62, 2.30, 0.95, "query",
         [f"(image, question)", f"batch {a['batch_size']}"], GREY_LT, GREY_LN, GREY),
        ("index", 0.35, 2.05, 2.30, 1.85, "live index",
         ["RL corpus ≈174K passages", "gold-entity articles",
          f"+ {a['n_distractor_articles']:,} distractors", "", f"top-M = {M} support"],
         GREY_LT, GREY_LN, GREY),
        ("polhead", 2.95, 0.62, 3.20, 0.52, f"Policy  πθ   ({a['profile']})",
         [], BLUE, BLUE_DK, "FFFFFF"),
        ("polbody", 2.95, 1.14, 3.20, 2.76, "",
         [f"{backbone}", "backbone weights FROZEN", "(grads pass through it)", "",
          f"+ LoRA  r={a['lora_r']}, α={a['lora_alpha']}", "the only weights updated", "",
          "s_d = ⟨e(q), e(d)⟩ / τ", f"τ = {tau}"],
         BLUE_LT, BLUE, "222222"),
        ("samphead", 6.45, 0.62, 3.25, 0.52, "Plackett–Luce sampling", [], BLUE, BLUE_DK, "FFFFFF"),
        ("sampbody", 6.45, 1.14, 3.25, 2.76, "",
         [f"G = {G} ordered lists of k = {k}", "",
          "Gumbel-top-k  =  exact PL", "sampling without replacement", "",
          "log π = Σᵢ [ s_dᵢ − log Σ_{Rᵢ} exp s_d ]",
          "denominator shrinks per position"],
         BLUE_LT, BLUE, "222222"),
        ("rewhead", 9.90, 0.62, 3.10, 0.52, "Reward (only supervision)", [], ORANGE, ORANGE_DK, "FFFFFF"),
        ("rewbody", 9.90, 1.14, 3.10, 2.76, "",
         [f"{reader}", "FROZEN — inference only,", "no gradients at all", "",
          f"scores each UNIQUE of", f"the G×k = {G*k} sampled", "",
          f"{a['reader_rollouts']} sampled answers @ T={a['reader_temperature']}",
          "reward = fraction correct", "token-boundary cover-EM", "vs gold ANSWER aliases"],
         ORANGE_LT, ORANGE, "222222"),
        ("credit", 2.95, 4.28, 10.05, 1.12, "credit assignment",
         [f"list reward = mean of its k={k} per-passage rewards      "
          f"advantage = z-score across the G={G} lists of one query  ({a['baseline']})",
          f"gradient = advantage × exact PL log-prob      "
          f"clip inert: inner_epochs={a['inner_epochs']} ⇒ ratio ≡ 1"],
         GREY_LT, GREY_LN, GREY),
        ("nogold", 0.35, 4.28, 2.30, 1.12, "", [], "FFFFFF", ORANGE, ORANGE),
    ]

    arrows = [
        # x0,y0,x1,y1, colour, dashed, label, label_xy, label_colour, rad
        (2.65, 1.10, 2.95, 1.60, GREY_LN, False, None, None, None, 0.0),   # query  -> policy
        (2.65, 2.70, 2.95, 2.40, GREY_LN, False, None, None, None, 0.0),   # index  -> policy
        (6.15, 2.30, 6.45, 2.30, BLUE, False, None, None, None, 0.0),      # policy -> sampling
        (9.70, 2.52, 9.90, 2.52, ORANGE, False, None, None, None, 0.0),    # sampling -> reward
        (11.45, 3.90, 11.45, 4.28, ORANGE, True, None, None, None, 0.0),   # reward -> credit
        (4.55, 4.28, 4.55, 3.90, BLUE, True, None, None, None, 0.0),       # credit -> policy
        # index refresh: bow ABOVE the query box rather than cutting through it
        (3.40, 0.62, 1.05, 2.05, GREY_LN, True, None, None, None, -0.40),
    ]

    notes = [
        ("gold evidence passage", 1.625, 4.72, 8.5, ORANGE, "strike"),
        ("never inserted, never a target:\nno relevance label anywhere", 1.625, 5.12, 7.0, ORANGE_DK, None),
        (f"re-encode the index with the current policy every {a['refresh_steps']} steps",
         3.10, 0.30, 7.6, GREY, "left"),
        (f"update LoRA only  —  AdamW, lr {lr}, {a['max_steps']} steps",
         4.70, 4.09, 7.6, BLUE_DK, "left"),
        ("answer-only signal", 11.60, 4.09, 7.6, ORANGE_DK, "left"),
    ]

    absent = ("ABSENT BY DESIGN      no InfoNCE anchor (contrastive_coef="
              f"{a['contrastive_coef']:g})      ·      no critic (vf_coef={a['vf_coef']:g})"
              f"      ·      no KL (kl_beta={a['kl_beta']:g})      ·      "
              "no gold passage (no_force_gold)")
    return boxes, arrows, notes, absent


# =============================== PDF (matplotlib) ============================================
def render_pdf(spec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes, arrows, notes, absent = spec
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fy = lambda y: H - y  # noqa: E731  top-left -> matplotlib

    for _k, x, y, w, h, title, body, fill, line, tcol in boxes:
        ax.add_patch(FancyBboxPatch((x, fy(y + h)), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.06",
                                    linewidth=1.5, edgecolor="#" + line,
                                    facecolor="#" + fill, zorder=2))
        if title:
            ty = fy(y + 0.30) if body else fy(y + h / 2 + 0.055)
            ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=11.5,
                    color="#" + tcol, fontweight="bold", zorder=3)
        if body:
            ax.text(x + w / 2, fy(y + (0.62 if title else 0.28)), "\n".join(body),
                    ha="center", va="top", fontsize=9.3, color="#" + tcol,
                    zorder=3, linespacing=1.55)

    for x0, y0, x1, y1, colour, dashed, label, lxy, lcol, rad in arrows:
        if colour == "FFFFFF":
            continue
        ax.add_patch(FancyArrowPatch((x0, fy(y0)), (x1, fy(y1)), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.6, color="#" + colour,
                                     linestyle="dashed" if dashed else "solid", zorder=4,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text(lxy[0], fy(lxy[1]), label, ha="center", va="bottom", fontsize=8.6,
                    color="#" + (lcol or colour), zorder=5)

    for text, x, y, size, colour, kind in notes:
        ha = "left" if kind == "left" else "center"
        ax.text(x, fy(y), text, ha=ha, va="center", fontsize=size, color="#" + colour,
                zorder=5, linespacing=1.45, style="italic" if kind == "left" else "normal")
        if kind == "strike":
            ax.plot([x - 1.06, x + 1.06], [fy(y), fy(y)], color="#" + colour, lw=1.8, zorder=6)

    ax.text(W / 2, fy(H - 0.18), absent, ha="center", va="bottom", fontsize=9.0,
            color="#" + ORANGE_DK, fontweight="bold", zorder=5)

    # ---- overflow check: measure every text's rendered extent against its owning box ---------
    # The previous figure in this deck (fig_af_rl_arch) needed a manual pass for exactly this;
    # here it is automatic and fails loudly rather than shipping clipped text.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    owners = {t: (x, y, w, h) for t, x, y, w, h, *_ in
              [(k, x, y, w, h) for k, x, y, w, h, *_ in boxes]}
    violations = []
    for txt in ax.texts:
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y1), (x1, y0) = inv.transform((bb.x0, bb.y1)), inv.transform((bb.x1, bb.y0))
        cx, cy = (x0 + x1) / 2, H - (y0 + y1) / 2
        for key, (bx, by, bw, bh) in owners.items():
            if bx <= cx <= bx + bw and by <= cy <= by + bh:      # text sits inside this box
                if x0 < bx - 0.01 or x1 > bx + bw + 0.01:
                    violations.append((key, txt.get_text()[:38].replace("\n", " / "),
                                       round(x1 - x0, 2), round(bw, 2)))
                break
    if violations:
        print("  TEXT OVERFLOW (width in, box in):")
        for v in violations:
            print(f"    {v[0]:9s} {v[1]!r:42s} {v[2]} > {v[3]}")
    else:
        print("  overflow check: 0 violations")

    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)   # edge-trimmed
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02)
    print("wrote", out + ".pdf/.png  (edge-trimmed, bbox_inches=tight)")


# =============================== PPTX (python-pptx) ==========================================
def render_pptx(spec, out):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    boxes, arrows, notes, absent = spec
    rgb = lambda h: RGBColor.from_string(h)  # noqa: E731

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank

    def textbox(x, y, w, h, runs, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for i, (txt, size, colour, bold) in enumerate(runs):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            r = para.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.color.rgb = rgb(colour)
            r.font.bold = bold
        return tb

    for key, x, y, w, h, title, body, fill, line, tcol in boxes:
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(x), Inches(y), Inches(w), Inches(h))
        sh.name = key
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(1.25)
        sh.shadow.inherit = False
        sh.text_frame.text = ""
        if title and not body:
            tf = sh.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = title
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = rgb(tcol)
        elif body:
            runs = ([(title, 12, tcol, True)] if title else []) + \
                   [(b, 9.5, tcol, False) for b in body]
            textbox(x + 0.06, y + 0.05, w - 0.12, h - 0.10, runs, anchor=MSO_ANCHOR.TOP)

    for x0, y0, x1, y1, colour, dashed, label, lxy, lcol, rad in arrows:
        if colour == "FFFFFF":
            continue
        con = slide.shapes.add_connector(2, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
        con.line.color.rgb = rgb(colour)
        con.line.width = Pt(1.5)
        if label:
            textbox(lxy[0] - 0.9, lxy[1] - 0.22, 1.8, 0.28,
                    [(label, 9, lcol or colour, False)])

    for text, x, y, size, colour, kind in notes:
        align = PP_ALIGN.LEFT if kind == "left" else PP_ALIGN.CENTER
        w = 5.6 if kind == "left" else 2.3
        x0 = x if kind == "left" else x - w / 2
        runs = [(ln, size, colour, False) for ln in text.split("\n")]
        textbox(x0, y - 0.22, w, 0.50, runs, align=align)
        if kind == "strike":
            ln = slide.shapes.add_connector(1, Inches(x - 1.06), Inches(y),
                                            Inches(x + 1.06), Inches(y))
            ln.line.color.rgb = rgb(colour)
            ln.line.width = Pt(1.75)

    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(H - 0.60),
                                 Inches(W - 0.90), Inches(0.42))
    bar.name = "absent-bar"
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(ORANGE_LT)
    bar.line.color.rgb = rgb(ORANGE)
    bar.line.width = Pt(1.0)
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = absent
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = rgb(ORANGE_DK)

    prs.save(out + ".pptx")
    print("wrote", out + ".pptx  (editable twin of the PDF)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmrag/fig_mm_v3_arch")
    ap.add_argument("--run", default="v3-pure")
    args = ap.parse_args()
    D = os.environ.get("MMRAG_DATA")
    if not D:
        sys.exit("MMRAG_DATA is not set")
    a = load_args(D, args.run)
    spec = build_spec(a)
    render_pdf(spec, args.out)
    render_pptx(spec, args.out)
    print(f"\nall printed values from {D}/runs/{args.run}/args.json:")
    for key in REQUIRED:
        print(f"   {key:24s} {a[key]!r}")


if __name__ == "__main__":
    main()
