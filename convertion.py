import json
import numpy as np
from tqdm import tqdm

SRC_META = "/media/shrish/Data/medpao_fast/snomed_index/snomed_metadata.jsonl"

EMB_META_OUT = "embedding_metadata.jsonl"
EMB_OFFSETS  = "embedding_offsets.npy"

CON_META_OUT = "concept_metadata.jsonl"
CON_OFFSETS  = "concept_offsets.npy"
CON_MAP_OUT  = "concept_id_to_index.json"


def convert():
    emb_offsets = []
    con_offsets = []
    concept_map = {}   # concept_id → index
    concept_seen = set()

    # open files
    emb_out = open(EMB_META_OUT, "w", encoding="utf-8")
    con_out = open(CON_META_OUT, "w", encoding="utf-8")

    emb_pos = 0
    con_pos = 0
    con_idx = 0

    print("Processing JSONL...")

    with open(SRC_META, encoding="utf-8") as f:
        for line in tqdm(f):
            row = json.loads(line)

            cid = row["concept_id"]

            # ── 1. Write embedding metadata (lightweight)
            emb_row = {
                "surface_form": row["surface_form"],
                "concept_id": cid
            }

            s = json.dumps(emb_row, ensure_ascii=False)
            emb_offsets.append(emb_pos)
            emb_out.write(s + "\n")
            emb_pos += len(s) + 1

            # ── 2. Write concept metadata (only once per concept)
            if cid not in concept_seen:
                concept_seen.add(cid)

                con_row = {
                    "concept_id": cid,
                    "preferred_desc": row["preferred_desc"],
                    "all_descs": row["all_descs"],
                    "ancestors": row["ancestors"]
                }

                s = json.dumps(con_row, ensure_ascii=False)
                con_offsets.append(con_pos)
                con_out.write(s + "\n")
                con_pos += len(s) + 1

                concept_map[cid] = con_idx
                con_idx += 1

    emb_out.close()
    con_out.close()

    # ── Save offsets + map
    np.save(EMB_OFFSETS, np.array(emb_offsets, dtype=np.int64))
    np.save(CON_OFFSETS, np.array(con_offsets, dtype=np.int64))

    with open(CON_MAP_OUT, "w") as f:
        json.dump(concept_map, f)

    print("\nDone.")
    print(f"Embedding rows : {len(emb_offsets):,}")
    print(f"Concept rows   : {len(con_offsets):,}")


if __name__ == "__main__":
    convert()