# ============================================================
# CELL 2: The matching tool  (load once, call many times)
# ============================================================

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

SAVE_DIR = './snomed_index'
EMB_FILE  = f'{SAVE_DIR}/snomed_embeddings.npy'
META_FILE = f'{SAVE_DIR}/snomed_metadata.jsonl'


class SnomedMatcher:
    """
    Lightweight retrieval tool that works entirely from the saved
    .npy embedding matrix and .jsonl metadata file — no SNOMED
    distribution or NetworkX graph required at query time.
    """

    def __init__(self, emb_file: str = EMB_FILE, meta_file: str = META_FILE,
                 device: str = 'cuda'):
        self.device = device

        # ── Load metadata ───────────────────────────────────
        print("Loading metadata …")
        self.metadata: list[dict] = []
        with open(meta_file, encoding='utf-8') as fh:
            for line in fh:
                self.metadata.append(json.loads(line))

        # ── Load embeddings onto GPU ────────────────────────
        print("Loading embeddings …")
        emb_np = np.load(emb_file)
        self.emb_gpu = torch.tensor(emb_np, dtype=torch.float32).to(device)
        print(f"  {self.emb_gpu.shape[0]:,} vectors ready on {device}")

        
        print("SnomedMatcher ready.\n")

    # ────────────────────────────────────────────────────────
    def match(self, queries: list[str], top_k: int = 1) -> list[list[dict]]:
        """
        Match a list of queries against the SNOMED embedding index.
        The encoder is unloaded from GPU after encoding to free memory
        for other models (e.g. LLMs) before the cdist computation.

        Parameters
        ----------
        queries : list of free-text clinical phrases
        top_k   : number of closest concepts per query (default 1)

        Returns
        -------
        List of lists — one inner list per query, each containing top_k dicts:
            concept_id, matched_surface, preferred_desc,
            all_descs, ancestors, distance
        """
        # ── Load SapBERT ────────────────────────────────────
        print("Loading SapBERT …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.model = AutoModel.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        ).to(self.device).eval()

        # ── Step 1: Encode all queries, then evict encoder from GPU ──────────
        with torch.no_grad():
            toks = self.tokenizer(
                queries,
                padding='max_length',
                max_length=25,
                truncation=True,
                return_tensors='pt',
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            q_vecs = self.model(**toks)[0][:, 0, :]   # (N_queries, hidden)

        # Move query vectors to CPU so cdist can run without encoder in VRAM
        q_vecs = q_vecs.cpu()

        # ── Step 2: Unload ONLY the encoder (keep embeddings + metadata) ─────
        self.model.cpu()
        del self.model
        # Also free the tokenizer's heavy vocab tensors if any
        del self.tokenizer
        torch.cuda.empty_cache()
        print("  [SnomedMatcher] Encoder unloaded from GPU.")

        # ── Step 3: cdist on CPU (encoder gone, VRAM now free for LLMs) ──────
        emb_cpu = self.emb_gpu.cpu()   # move index to CPU just for this op
        dists = torch.cdist(q_vecs, emb_cpu)   # (N_queries, N_concepts)

        # ── Step 4: Collect top_k results per query ───────────────────────────
        all_results: list[list[dict]] = []
        for i, query in enumerate(queries):
            top_vals, top_idx = torch.topk(dists[i], k=top_k, largest=False)
            query_results = []
            for dist_val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                row = self.metadata[idx]
                query_results.append({
                    'query'           : query,
                    'concept_id'      : row['concept_id'],
                    'matched_surface' : row['surface_form'],
                    'preferred_desc'  : row['preferred_desc'],
                    'all_descs'       : row['all_descs'],
                    'ancestors'       : row['ancestors'],
                    'distance'        : round(dist_val, 6),
                })
            all_results.append(query_results)

        return all_results