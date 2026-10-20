"""Where does the slate actually go when the index is a 2M mixed-domain haystack?

The 2M index is not a uniform blur of distractors: built/corpus_mix2M.composition.json records that it
is corpus_div_v2 (Wikipedia + OVEN + OK-VQA, carried byte-identically so every gold keeps its pid)
topped up with nine BEIR domains and HotpotQA. Every passage's `url` carries its provenance
(`beir://trec-covid/...`, `oven://Q...`, `https://en.wikipedia.org/...`), so the top-k pids already
stored in results/*-on-mix2M.retrieval.json can be attributed to a source domain with no GPU at all.

WHAT THIS BUYS. The paper's haystack result is a LEVEL: entR@5 falls 68.53 -> 14.27 when the index grows
from 422k in-domain passages to 2M mixed ones. A level says the slate got worse; it does not say which
haystack did it. This attributes the damage: a domain that is 12.3% of the index and 12.3% of the
returned slate is inert volume, while one that is 4.5% of the index and 15% of the slate is an actual
attractor. LIFT = share of slate / share of index; lift 1.0 is "no worse than its size".

CAVEAT, stated because it is easy to over-read. A retrieved off-domain passage is not automatically an
error — it is only an error if it displaced a gold. So the per-domain table is reported alongside the
arm's entR@5 on the same index, and the comparison that matters is BETWEEN ARMS on the same domain, not
the absolute share of any one domain.

    python3 mmrag/haystack_domains.py --k 5
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

D = os.environ.get("MMRAG_DATA", "/raid/icy/mmrag_data")
R = os.path.join(D, "results")
MAP = os.path.join(D, "built", "corpus_mix2M.domains.json")
LEAD = os.path.join(D, "built", "corpus_mix2M.lead.json")


def domain_of(url):
    if not url:
        return "wiki"                                   # base div pool rows without a url
    m = re.match(r"beir://([^/]+)/", url)
    if m:
        return m.group(1)
    if url.startswith("oven://"):
        return "oven"
    if url.startswith("okvqa://"):
        return "okvqa"
    if "wikipedia.org" in url:
        return "wiki"
    return "other"


def build_map(corpus="corpus_mix2M.jsonl", mp=MAP, lp=LEAD):
    """pid -> (domain, is_lead), cached. One streaming pass reading pid, url and section only.

    is_lead marks a passage with no section heading: an entity LEAD/summary paragraph. Every OVEN row is
    one of these; InfoSeek gold evidence almost never is, because the questions ask about facts that live
    in mid-article sections. Separating format from domain is the whole point — "the base prefers OVEN"
    and "the base prefers lead paragraphs" predict the same 2M table but different in-domain behaviour."""
    if os.path.exists(mp) and os.path.exists(lp):
        return ({int(k): v for k, v in json.load(open(mp)).items()},
                {int(k): v for k, v in json.load(open(lp)).items()})
    dom, lead = {}, {}
    with open(os.path.join(D, "built", corpus)) as f:
        for line in f:
            r = json.loads(line)
            pid = int(r["pid"])
            dom[pid] = domain_of(r.get("url"))
            lead[pid] = int(not (r.get("section") or "").strip())
    json.dump(dom, open(mp, "w"))
    json.dump(lead, open(lp, "w"))
    return dom, lead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min_index_share", type=float, default=0.15,
                    help="hide domains below this %% of the index; they cannot move a table")
    args = ap.parse_args()

    pid2dom, pid2lead = build_map()
    n_index = len(pid2dom)
    idx_share = Counter(pid2dom.values())

    arms = sorted(f[:-len(".retrieval.json")] for f in os.listdir(R)
                  if f.endswith("-on-mix2M.retrieval.json"))
    if not arms:
        print("no *-on-mix2M.retrieval.json yet")
        return

    rows, ent, lead_share = {}, {}, {}
    for arm in arms:
        topk = json.load(open(os.path.join(R, f"{arm}.retrieval.json")))
        c = Counter()
        tot = 0
        for qid, d in topk.items():
            for p in d["topk"][: args.k]:
                c[pid2dom.get(int(p), "other")] += 1
                tot += 1
        rows[arm] = {k: 100.0 * v / tot for k, v in c.items()}
        nlead = sum(pid2lead.get(int(p), 0) for d in topk.values() for p in d["topk"][: args.k])
        lead_share[arm] = 100.0 * nlead / tot
        mp = os.path.join(R, f"{arm}.retrieval.metrics.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            ent[arm] = 100.0 * m["entity_recall"]["5"]

    doms = [d for d, n in idx_share.most_common()
            if 100.0 * n / n_index >= args.min_index_share]

    print(f"\n  index: {n_index:,} passages   arms scored on it: {len(arms)}   slate depth k={args.k}\n")
    print(f"  {'domain':16s} {'%index':>8s}   " + "  ".join(f"{a.replace('-on-mix2M',''):>14s}" for a in arms))
    print(f"  {'':16s} {'':>8s}   " + "  ".join(f"{'%slate / lift':>14s}" for a in arms))
    for d in doms:
        si = 100.0 * idx_share[d] / n_index
        cells = []
        for a in arms:
            ss = rows[a].get(d, 0.0)
            cells.append(f"{ss:6.2f} /{ss/si:5.2f}")
        print(f"  {d:16s} {si:7.2f}%   " + "  ".join(f"{c:>14s}" for c in cells))
    print(f"\n  {'entR@5 on this index':25s} " + "  ".join(
        f"{(f'{ent[a]:.2f}' if ent.get(a) is not None else '--'):>14s}" for a in arms))
    bl = 100.0 * sum(pid2lead.values()) / n_index
    print(f"\n  LEAD-PARAGRAPH SHARE OF THE SLATE  (index baseline {bl:.2f}%)")
    for a in arms:
        print(f"    {a.replace('-on-mix2M',''):25s} {lead_share[a]:6.2f}%  lift {lead_share[a]/bl:5.2f}")
    offw = sum(100.0 * idx_share[d] / n_index for d in idx_share if d not in ("wiki", "oven", "okvqa"))
    print(f"  {'off-domain share of index':25s} {offw:6.2f}%")
    for a in arms:
        off = sum(v for d, v in rows[a].items() if d not in ("wiki", "oven", "okvqa"))
        print(f"  {'  ' + a.replace('-on-mix2M',''):25s} off-domain share of slate {off:6.2f}%  lift {off/offw:5.2f}")


if __name__ == "__main__":
    main()
