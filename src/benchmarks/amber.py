"""AMBER generative-task matcher (LLM-free), for the v_c hallucination diagnostic.

AMBER (Wang et al.) ships per-image annotations for the generative task:
  truth : objects actually present in the image
  hallu : plausible-but-absent objects (the curated hallucination targets)
and ``relation.json`` mapping a canonical object to its synonyms.

For our object-level v_c test we only need, per generated caption, the mentioned
objects that are *explicitly annotated* as either present (truth -> grounded) or
a hallucination target (hallu -> hallucinated).  Restricting to truth ∪ hallu
gives clean, annotation-grounded labels without spaCy word-vectors (AMBER's full
metric uses those; we only need the grounded/hallucinated label per object).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk
from nltk.stem import WordNetLemmatizer

GEN_ID_RANGE = (1, 1004)


class AMBERGen:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        ann = json.loads((data_dir / "annotations.json").read_text())
        rel = json.loads((data_dir / "relation.json").read_text())
        # per-image truth / hallu for generative ids
        self.truth: Dict[int, Set[str]] = {}
        self.hallu: Dict[int, Set[str]] = {}
        for a in ann:
            if a.get("type") == "generative":
                self.truth[a["id"]] = set(a.get("truth", []))
                self.hallu[a["id"]] = set(a.get("hallu", []))
        # Synonym GROUPS via union-find: relation.json lists {word: [synonyms]}
        # but truth/hallu use specific members, so the map must be SYMMETRIC
        # (forest <-> bush must land in the same group regardless of direction).
        parent: Dict[str, str] = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            parent[find(a)] = find(b)
        for k, syns in rel.items():
            find(k)
            for s in syns:
                union(k, s)
        for d in (self.truth, self.hallu):       # standalone truth/hallu words
            for s in d.values():
                for w in s:
                    find(w)
        self._find = find
        self.vocab = set(parent.keys())
        # precompute component ids of each image's truth/hallu sets
        self._truth_comp = {i: {find(w) for w in s} for i, s in self.truth.items()}
        self._hallu_comp = {i: {find(w) for w in s} for i, s in self.hallu.items()}
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        self._lem = WordNetLemmatizer()

    def singular(self, w: str) -> str:
        return self._lem.lemmatize(w.lower(), pos="n")

    def objects_in_caption(self, caption: str, image_id: int
                           ) -> List[Tuple[str, Tuple[int, int], bool]]:
        """Return [(canonical, (char_start, char_end), hallucinated), ...] for
        objects that are annotated (in truth or hallu) for this image.

        Maps each surface word to a canonical AMBER object via the relation
        synonym table; keeps only objects explicitly labelled present (grounded)
        or hallucination-target (hallucinated)."""
        tcomp = self._truth_comp.get(image_id, set())
        hcomp = self._hallu_comp.get(image_id, set())
        low = caption.lower()
        out = []
        for m in re.finditer(r"[a-zA-Z]+", low):
            w = self.singular(m.group())
            if w not in self.vocab:
                continue
            c = self._find(w)
            if c in tcomp:                       # present (grounded) takes priority
                out.append((w, (m.start(), m.end()), False))
            elif c in hcomp:                     # plausible-but-absent (hallucinated)
                out.append((w, (m.start(), m.end()), True))
        return out
