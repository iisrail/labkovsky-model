#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedupe review pipeline for card_principals.jsonl.

Creates clusters of similar principles for manual review.
Does NOT automatically delete or merge anything.

Output:
1. principle_dedupe_clusters.jsonl - structured clusters
2. principle_dedupe_report.txt - human-readable report
"""

import json
import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
INPUT_FILE = Path("data/fine_tuning/card_principals.jsonl")
OUTPUT_CLUSTERS = Path("data/fine_tuning/principle_dedupe_clusters.jsonl")
OUTPUT_REPORT = Path("data/fine_tuning/principle_dedupe_report.txt")

# Similarity thresholds
THRESH_DUPLICATE = 0.92      # >= 0.92: probable duplicate
THRESH_MERGE = 0.86          # 0.86-0.92: merge candidate
THRESH_RELATED = 0.78        # 0.78-0.86: related, review manually
# < 0.78: ignore

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# DT normalization map
DT_NORMALIZE = {
    "EXPLANATION": "EXPLANATION",
    "INTERVENTION": "INTERVENTION",
    "ESCALATION": "CLINICAL_ESCALATION",
    "CLINICAL_ESCALATION": "CLINICAL_ESCALATION",
    "SELF_ESTEEM_CORRECTIVE": "SELF_ESTEEM_CORRECTIVE",
    "DEPENDENCY_BOUNDARIES": "DEPENDENCY_BOUNDARIES",
    "ANXIETY_MANAGEMENT": "ANXIETY_MANAGEMENT",
    "AFFECTIVE_ADDICTION": "AFFECTIVE_ADDICTION",
    "ADDICTION_PATTERN": "ADDICTION_PATTERN",
    "PARENTING_MODEL": "PARENTING_MODEL",
    "PARENTING_LIMITS": "PARENTING_LIMITS",
    "FEAR_SCENARIO_COPING": "FEAR_SCENARIO_COPING",
}


def load_records(filepath: Path) -> List[Dict[str, Any]]:
    """Load records with line numbers."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    rec['_line'] = idx + 1
                    rec['_idx'] = idx
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def normalize_dt(dt: str) -> str:
    """Normalize DT name."""
    return DT_NORMALIZE.get(dt, dt)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_action_suggestion(sim: float, same_dt: bool) -> str:
    """Get action suggestion based on similarity and DT match."""
    if sim >= THRESH_DUPLICATE:
        return "DUPLICATE"
    elif sim >= THRESH_MERGE:
        if same_dt:
            return "MERGE"
        else:
            return "RELATED_KEEP_SEPARATE"
    elif sim >= THRESH_RELATED:
        return "RELATED_KEEP_SEPARATE"
    else:
        return "IGNORE"


def find_clusters(
    records: List[Dict],
    embeddings: np.ndarray
) -> List[Dict[str, Any]]:
    """Find clusters of similar principles."""
    n = len(records)
    used = set()
    clusters = []
    cluster_id = 0

    # Group by DT first for efficiency
    dt_groups = defaultdict(list)
    for i, rec in enumerate(records):
        dt = normalize_dt(rec.get('dt', 'UNKNOWN'))
        dt_groups[dt].append(i)

    print(f"Processing {n} records...")

    # First pass: within same DT
    for dt, indices in dt_groups.items():
        print(f"  Processing DT: {dt} ({len(indices)} records)")

        for i_pos, i in enumerate(indices):
            if i in used:
                continue

            cluster_members = []

            for j in indices[i_pos + 1:]:
                if j in used:
                    continue

                sim = cosine_similarity(embeddings[i], embeddings[j])

                if sim >= THRESH_RELATED:
                    action = get_action_suggestion(sim, same_dt=True)
                    if action != "IGNORE":
                        cluster_members.append({
                            'idx': j,
                            'similarity': sim,
                            'action': action,
                        })

            if cluster_members:
                # Sort by similarity descending
                cluster_members.sort(key=lambda x: x['similarity'], reverse=True)

                # Mark as used
                used.add(i)
                for m in cluster_members:
                    used.add(m['idx'])

                # Build cluster
                cluster = build_cluster(
                    cluster_id=cluster_id,
                    anchor_idx=i,
                    members=cluster_members,
                    records=records,
                )
                clusters.append(cluster)
                cluster_id += 1

    # Second pass: cross-DT for very high similarity only
    print("  Cross-DT pass for high similarity...")
    remaining = [i for i in range(n) if i not in used]

    for i_pos, i in enumerate(remaining):
        if i in used:
            continue

        cluster_members = []

        for j in remaining[i_pos + 1:]:
            if j in used:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            # Only include cross-DT if very high similarity
            if sim >= THRESH_DUPLICATE:
                dt_i = normalize_dt(records[i].get('dt', 'UNKNOWN'))
                dt_j = normalize_dt(records[j].get('dt', 'UNKNOWN'))
                same_dt = dt_i == dt_j

                action = get_action_suggestion(sim, same_dt)
                if action != "IGNORE":
                    cluster_members.append({
                        'idx': j,
                        'similarity': sim,
                        'action': action,
                    })

        if cluster_members:
            cluster_members.sort(key=lambda x: x['similarity'], reverse=True)

            used.add(i)
            for m in cluster_members:
                used.add(m['idx'])

            cluster = build_cluster(
                cluster_id=cluster_id,
                anchor_idx=i,
                members=cluster_members,
                records=records,
            )
            clusters.append(cluster)
            cluster_id += 1

    return clusters


def build_cluster(
    cluster_id: int,
    anchor_idx: int,
    members: List[Dict],
    records: List[Dict],
) -> Dict[str, Any]:
    """Build a cluster record."""
    anchor = records[anchor_idx]

    # Collect all records in cluster
    all_indices = [anchor_idx] + [m['idx'] for m in members]
    all_records = []

    for idx in all_indices:
        rec = records[idx]
        all_records.append({
            'line': rec.get('_line'),
            'idx': rec.get('_idx'),
            'source': rec.get('source', ''),
            'book': rec.get('book', ''),
            'chapter': rec.get('chapter', ''),
            'qa_id': rec.get('qa_id', rec.get('chapter', '')),
            'dt': normalize_dt(rec.get('dt', 'UNKNOWN')),
            'core_principle': rec.get('core_principle', ''),
        })

    # Get all DTs
    dts = list(set(r['dt'] for r in all_records))

    # Determine overall action
    if members:
        max_sim = max(m['similarity'] for m in members)
        if max_sim >= THRESH_DUPLICATE:
            action = "DUPLICATE"
        elif max_sim >= THRESH_MERGE and len(dts) == 1:
            action = "MERGE"
        else:
            action = "RELATED_KEEP_SEPARATE"
    else:
        action = "RELATED_KEEP_SEPARATE"

    # Suggest canonical (shortest clear one, or first)
    principles = [r['core_principle'] for r in all_records]
    # Prefer medium length (not too short, not too long)
    scored = [(len(p), i, p) for i, p in enumerate(principles)]
    scored.sort(key=lambda x: abs(x[0] - 150))  # Prefer ~150 chars
    suggested_canonical = scored[0][2]

    # Build reason
    if action == "DUPLICATE":
        reason = f"High similarity ({max_sim:.3f}), likely exact or near-exact duplicate"
    elif action == "MERGE":
        reason = f"Similar content ({max_sim:.3f}), same DT, could merge into one principle"
    else:
        reason = f"Related but distinct ({max_sim:.3f}), review manually"

    return {
        'cluster_id': cluster_id,
        'dt_values': dts,
        'action_suggestion': action,
        'max_similarity': max_sim if members else 0,
        'records': all_records,
        'suggested_canonical_core_principle': suggested_canonical,
        'reason': reason,
    }


def write_report(clusters: List[Dict], filepath: Path):
    """Write human-readable report."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PRINCIPLE DEDUPLICATION REVIEW REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Stats
        duplicates = [c for c in clusters if c['action_suggestion'] == 'DUPLICATE']
        merges = [c for c in clusters if c['action_suggestion'] == 'MERGE']
        related = [c for c in clusters if c['action_suggestion'] == 'RELATED_KEEP_SEPARATE']

        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"  DUPLICATE: {len(duplicates)}\n")
        f.write(f"  MERGE: {len(merges)}\n")
        f.write(f"  RELATED_KEEP_SEPARATE: {len(related)}\n\n")

        # By action
        for action_name, action_clusters in [
            ("DUPLICATE", duplicates),
            ("MERGE", merges),
            ("RELATED_KEEP_SEPARATE", related),
        ]:
            if not action_clusters:
                continue

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{action_name} CLUSTERS ({len(action_clusters)})\n")
            f.write("=" * 80 + "\n")

            for cluster in action_clusters:
                f.write(f"\n--- Cluster {cluster['cluster_id']} ---\n")
                f.write(f"DT: {', '.join(cluster['dt_values'])}\n")
                f.write(f"Similarity: {cluster['max_similarity']:.3f}\n")
                f.write(f"Reason: {cluster['reason']}\n")
                f.write(f"Suggested canonical:\n  {cluster['suggested_canonical_core_principle'][:200]}...\n")
                f.write("\nRecords:\n")

                for rec in cluster['records']:
                    f.write(f"\n  [{rec['line']}] {rec['source']} | {rec.get('book', '')[:20]} | Ch:{rec.get('chapter', '')[:10]}\n")
                    f.write(f"  DT: {rec['dt']}\n")
                    f.write(f"  {rec['core_principle'][:300]}...\n")

                f.write("\n")


def main():
    print(f"Loading records from {INPUT_FILE}...")
    records = load_records(INPUT_FILE)
    print(f"Loaded {len(records)} records")

    # Normalize DTs
    for rec in records:
        rec['dt'] = normalize_dt(rec.get('dt', 'UNKNOWN'))

    # DT distribution
    dt_counts = defaultdict(int)
    for rec in records:
        dt_counts[rec['dt']] += 1
    print("\nDT distribution:")
    for dt, count in sorted(dt_counts.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    print("Computing embeddings...")
    texts = [f"passage: {r.get('core_principle', '')}" for r in records]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Computed {len(embeddings)} embeddings")

    # Find clusters
    print("\nFinding clusters...")
    clusters = find_clusters(records, embeddings)
    print(f"Found {len(clusters)} clusters")

    # Stats
    duplicates = sum(1 for c in clusters if c['action_suggestion'] == 'DUPLICATE')
    merges = sum(1 for c in clusters if c['action_suggestion'] == 'MERGE')
    related = sum(1 for c in clusters if c['action_suggestion'] == 'RELATED_KEEP_SEPARATE')

    print(f"\nCluster breakdown:")
    print(f"  DUPLICATE: {duplicates}")
    print(f"  MERGE: {merges}")
    print(f"  RELATED_KEEP_SEPARATE: {related}")

    # Records in clusters
    records_in_clusters = sum(len(c['records']) for c in clusters)
    print(f"\nRecords in clusters: {records_in_clusters}")
    print(f"Records not in clusters: {len(records) - records_in_clusters}")

    # Save clusters
    print(f"\nSaving clusters to {OUTPUT_CLUSTERS}...")
    with open(OUTPUT_CLUSTERS, 'w', encoding='utf-8') as f:
        for cluster in clusters:
            f.write(json.dumps(cluster, ensure_ascii=False) + '\n')

    # Write report
    print(f"Writing report to {OUTPUT_REPORT}...")
    write_report(clusters, OUTPUT_REPORT)

    print("\nDone!")
    print(f"Review the files:")
    print(f"  1. {OUTPUT_CLUSTERS} - structured data")
    print(f"  2. {OUTPUT_REPORT} - human-readable report")


if __name__ == "__main__":
    main()
