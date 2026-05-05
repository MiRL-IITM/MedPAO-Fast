import json
import numpy as np
import torch
from llama_cpp import Llama

SAVE_DIR = './snomed_index'

EMB_FILE = f'{SAVE_DIR}/snomed_embeddings.npy'

EMB_META_FILE = f'{SAVE_DIR}/embedding_metadata.jsonl'
EMB_OFFSETS   = f'{SAVE_DIR}/embedding_offsets.npy'

CONCEPT_META_FILE = f'{SAVE_DIR}/concept_metadata.jsonl'
CONCEPT_OFFSETS   = f'{SAVE_DIR}/concept_offsets.npy'
CONCEPT_ID_MAP    = f'{SAVE_DIR}/concept_id_to_index.json'

SAPBERT_GGUF = '/media/shrish/Data/medpao_fast/cpu_codes/gguf_models/sapbert.gguf'


class SnomedMatcher:

    def __init__(self,
                 emb_file=EMB_FILE,
                 gguf_path=SAPBERT_GGUF,
                 n_gpu_layers=0):

        self.gguf_path = gguf_path
        self.n_gpu_layers = n_gpu_layers

        # ── Load embeddings (NO COPY) ────────────────────────
        print("Loading embeddings (mmap) …")
        emb_np = np.load(emb_file, mmap_mode='r')
        self.emb = torch.from_numpy(emb_np).half()
        print(f"  {self.emb.shape[0]:,} vectors | dim={self.emb.shape[1]}")

        # ── Load offset indices (tiny memory) ────────────────
        print("Building offset indices …")
        self.emb_offsets = self._build_offsets(EMB_META_FILE)
        self.con_offsets = self._build_offsets(CONCEPT_META_FILE)
        print(f"  emb offsets: {len(self.emb_offsets):,} | con offsets: {len(self.con_offsets):,}")

        # concept_id → row index mapping
        with open(CONCEPT_ID_MAP) as f:
            self.cid_to_idx = json.load(f)

        # ── Open files once (important) ──────────────────────
        self.emb_meta_fh = open(EMB_META_FILE, 'rb')
        self.con_meta_fh = open(CONCEPT_META_FILE, 'rb')

        # ── Load encoder ONCE (critical fix) ─────────────────
        print("Loading SapBERT GGUF …")
        self.model = Llama(
            model_path=self.gguf_path,
            embedding=True,
            n_ctx=512,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

        print("SnomedMatcher ready.\n")

    def _build_offsets(self, filepath: str) -> np.ndarray:
        """Scan a JSONL file and record the byte offset of every line."""
        offsets = []
        with open(filepath, 'rb') as f:
            while True:
                offsets.append(f.tell())
                line = f.readline()
                if not line:
                    offsets.pop()  # remove the trailing EOF offset
                    break
        return np.array(offsets, dtype=np.int64)
    # ────────────────────────────────────────────────────────
    def __del__(self):
        self.emb_meta_fh.close()
        self.con_meta_fh.close()

    def _get_emb_meta(self, idx: int):
        offset = int(self.emb_offsets[idx])
        self.emb_meta_fh.seek(offset)
        raw = self.emb_meta_fh.readline()
        
        print(f"idx={idx} | offset={offset} | raw bytes preview: {raw[:80]}")
        
        return json.loads(raw.decode('utf-8'))

    def _get_concept_meta(self, concept_id: str):
        idx = self.cid_to_idx[concept_id]
        self.con_meta_fh.seek(int(self.con_offsets[idx]))
        return json.loads(self.con_meta_fh.readline().decode('utf-8'))

    # ────────────────────────────────────────────────────────
    def _encode(self, queries: list[str]) -> torch.Tensor:
        print(queries)
        vecs = []
        for q in queries:
            res = self.model.create_embedding(q)
            token_embs = np.array(res["data"][0]["embedding"]) 
            cls_vec = token_embs.mean(axis=0)
        
            vecs.append(cls_vec)

        return torch.tensor(vecs, dtype=torch.float16)

    # ────────────────────────────────────────────────────────
    def match(self, queries: list[str], top_k: int = 1):

        q_vecs = self._encode(queries)


        # distance computation
        dists = torch.cdist(q_vecs, self.emb)

        results = []
        for i, query in enumerate(queries):

            top_vals, top_idx = torch.topk(
                dists[i], k=top_k, largest=False
            )

            q_results = []
            for dist_val, idx in zip(top_vals.tolist(), top_idx.tolist()):

                # ── Step 1: get concept_id (cheap)
                emb_row = self._get_emb_meta(idx)
                cid = emb_row["concept_id"]

                # ── Step 2: get full concept metadata (lazy)
                concept = self._get_concept_meta(cid)

                q_results.append({
                    "query": query,
                    "concept_id": cid,
                    "matched_surface": emb_row["surface_form"],
                    "preferred_desc": concept["preferred_desc"],
                    "all_descs": concept["all_descs"],
                    "ancestors": concept["ancestors"],
                    "distance": round(dist_val, 6),
                })

            results.append(q_results)

        return results