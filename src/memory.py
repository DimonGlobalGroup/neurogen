"""
NEUROGEN Memory — associative and key-value variants.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math


class AssociativeMemory:
    """
    Simple associative memory prototype.
    Stores (key, value, tags) and supports similarity-based retrieval.
    """

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._index: Dict[str, List[int]] = defaultdict(list)  # tag -> indices

    def store(self, key: str, value: Any, tags: Optional[List[str]] = None, strength: float = 1.0):
        tags = tags or []
        entry = {
            "key": key,
            "value": value,
            "tags": set(tags),
            "strength": strength,
            "access_count": 0,
        }
        idx = len(self.entries)
        self.entries.append(entry)
        for t in tags:
            self._index[t].append(idx)
        self._index[key].append(idx)

    def retrieve(self, query: str, tags: Optional[List[str]] = None, top_k: int = 3) -> List[Tuple[Any, float]]:
        """Return list of (value, score) ranked by relevance."""
        tags = set(tags or [])
        scores = []
        for i, e in enumerate(self.entries):
            score = 0.0
            if e["key"] == query:
                score += 2.0
            if query in e["key"] or e["key"] in query:
                score += 1.0
            overlap = len(tags & e["tags"])
            score += overlap * 0.8
            score *= e["strength"]
            score += math.log1p(e["access_count"]) * 0.1
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        results = []
        for i, s in scores[:top_k]:
            self.entries[i]["access_count"] += 1
            results.append((self.entries[i]["value"], s))
        return results

    def strengthen(self, key: str, amount: float = 0.1):
        for e in self.entries:
            if e["key"] == key:
                e["strength"] = min(5.0, e["strength"] + amount)

    def weaken(self, key: str, amount: float = 0.1):
        for e in self.entries:
            if e["key"] == key:
                e["strength"] = max(0.0, e["strength"] - amount)

    def dump(self) -> List[Dict]:
        return [
            {
                "key": e["key"],
                "value": e["value"],
                "tags": list(e["tags"]),
                "strength": e["strength"],
                "access_count": e["access_count"],
            }
            for e in self.entries
        ]


class KeyValueMemory:
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.store[key] = value

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def dump(self):
        return dict(self.store)
