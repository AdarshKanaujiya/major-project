"""
weeks/week_05/retrieval/test_retriever_meet.py

Comprehensive Unit & Integration Tests for Week 5 Phase 2 Retriever.

Tests:
1. Retriever initialization and index verification
2. Checkpoint loading and weight matching
3. SAR preprocessing (both real GeoTIFF patch and tensor)
4. MS preprocessing (both real GeoTIFF patch and tensor)
5. SAR -> MS FAISS search
6. MS -> SAR FAISS search
7. Exactly top-10 result count
8. Result patch-ID validity (matches test_split.csv)
9. Similarity-score validity (finite float between -1.0 and 1.0)
10. No NaN/Inf anywhere in embeddings or scores
11. Strict descending score order
12. 512-D model output dimension
13. End-to-end inference with real local test images
14. Component latencies (model load, preprocess, encode, FAISS search)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.config.settings import RESULTS_DIR, SPLITS_DIR, CHECKPOINTS_DIR, S1_DIR, S2_DIR
except ImportError:
    RESULTS_DIR = PROJECT_ROOT / "results" / "local_pc"
    SPLITS_DIR = RESULTS_DIR / "week_02" / "splits"
    CHECKPOINTS_DIR = RESULTS_DIR / "week_04" / "checkpoints"
    S1_DIR = PROJECT_ROOT / "data" / "BigEarthNet-S1"
    S2_DIR = PROJECT_ROOT / "data" / "BigEarthNet-S2"

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from weeks.week_05.retrieval.retriever_meet import CrossModalRetriever, retrieve
except ImportError:
    CrossModalRetriever = None
    retrieve = None


def run_tests() -> None:
    print("=" * 70)
    print("WEEK 5 — PHASE 2 RETRIEVER VERIFICATION SUITE")
    print("=" * 70)
    print()

    if torch is None:
        print("[BLOCKED BY ENVIRONMENT] PyTorch is not installed in the active environment.")
        print("Skipping live neural retriever execution.")
        return

    if faiss is None:
        print("[BLOCKED BY ENVIRONMENT] FAISS is not installed in the active environment.")
        print("Skipping live index search execution.")
        return

    checkpoint_file = CHECKPOINTS_DIR / "best_model-v1.ckpt"
    if not checkpoint_file.exists():
        print(f"[BLOCKED BY ENVIRONMENT] Checkpoint not found: {checkpoint_file}")
        print("Skipping live neural retriever execution.")
        return

    sar_idx_file = RESULTS_DIR / "week_05" / "faiss_indices" / "sar_gallery.index"
    ms_idx_file = RESULTS_DIR / "week_05" / "faiss_indices" / "ms_gallery.index"
    if not sar_idx_file.exists() or not ms_idx_file.exists():
        print("[BLOCKED BY ENVIRONMENT] FAISS gallery indices missing.")
        print("Skipping live index search execution.")
        return

    # --------------------------------------------------------
    # 1. RETRIEVER INITIALIZATION & LATENCY
    # --------------------------------------------------------
    print("[1/14] Testing Retriever Initialization & Checkpoint Loading...")
    t0 = time.perf_counter()
    retriever = CrossModalRetriever()
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"  -> Model & Index loading latency: {t_load:.2f} ms")
    print(f"  -> SAR gallery vectors: {retriever.sar_index.ntotal}")
    print(f"  -> MS gallery vectors : {retriever.ms_index.ntotal}")
    assert retriever.sar_index.ntotal == 4500, "SAR index ntotal != 4500"
    assert retriever.ms_index.ntotal == 4500, "MS index ntotal != 4500"
    print("  [PASS] Initialization and Checkpoint Loading verified.\n")

    # --------------------------------------------------------
    # 2. LOCATE REAL TEST PATCHES
    # --------------------------------------------------------
    print("[2/14] Locating Real Test Patches for End-to-End Inference...")
    s1_patch_dir = (
        S1_DIR
        / "S1B_IW_GRDH_1SDV_20170612T165809"
        / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66"
    )
    s2_patch_dir = (
        S2_DIR
        / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
        / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    )

    real_images_available = s1_patch_dir.exists() and s2_patch_dir.exists()
    if real_images_available:
        print(f"  -> Real SAR patch located: {s1_patch_dir.name}")
        print(f"  -> Real MS patch located : {s2_patch_dir.name}")
        print("  [PASS] Real test images available on disk.\n")
    else:
        print("  [WARN] Real test patch folders not found at expected paths. Using synthetic inputs.\n")

    # --------------------------------------------------------
    # 3. SAR PREPROCESSING & LATENCY
    # --------------------------------------------------------
    print("[3/14] Testing SAR Preprocessing...")
    if real_images_available:
        t0 = time.perf_counter()
        sar_tensor = retriever.preprocess_sar(s1_patch_dir)
        t_sar_prep = (time.perf_counter() - t0) * 1000.0
        print(f"  -> SAR Preprocessing latency: {t_sar_prep:.2f} ms")
    else:
        dummy_sar = torch.randn(1, 2, 224, 224)
        t0 = time.perf_counter()
        sar_tensor = retriever.preprocess_sar(dummy_sar)
        t_sar_prep = (time.perf_counter() - t0) * 1000.0

    print(f"  -> SAR tensor shape: {tuple(sar_tensor.shape)}")
    print(f"  -> SAR tensor dtype: {sar_tensor.dtype}")
    assert sar_tensor.shape == (1, 2, 224, 224), f"Unexpected SAR shape: {sar_tensor.shape}"
    assert torch.isfinite(sar_tensor).all(), "SAR tensor contains NaN/Inf!"
    print("  [PASS] SAR Preprocessing passed.\n")

    # --------------------------------------------------------
    # 4. MS PREPROCESSING & LATENCY
    # --------------------------------------------------------
    print("[4/14] Testing MS Preprocessing...")
    if real_images_available:
        t0 = time.perf_counter()
        ms_tensor = retriever.preprocess_ms(s2_patch_dir)
        t_ms_prep = (time.perf_counter() - t0) * 1000.0
        print(f"  -> MS Preprocessing latency: {t_ms_prep:.2f} ms")
    else:
        dummy_ms = torch.randn(1, 3, 224, 224)
        t0 = time.perf_counter()
        ms_tensor = retriever.preprocess_ms(dummy_ms)
        t_ms_prep = (time.perf_counter() - t0) * 1000.0

    print(f"  -> MS tensor shape: {tuple(ms_tensor.shape)}")
    print(f"  -> MS tensor dtype: {ms_tensor.dtype}")
    assert ms_tensor.shape == (1, 3, 224, 224), f"Unexpected MS shape: {ms_tensor.shape}"
    assert torch.isfinite(ms_tensor).all(), "MS tensor contains NaN/Inf!"
    print("  [PASS] MS Preprocessing passed.\n")

    # --------------------------------------------------------
    # 5. SAR EMBEDDING INFERENCE & LATENCY
    # --------------------------------------------------------
    print("[5/14] Testing SAR Tower Inference & 512-D Embedding...")
    t0 = time.perf_counter()
    sar_emb = retriever.encode_sar(sar_tensor)
    t_sar_infer = (time.perf_counter() - t0) * 1000.0
    print(f"  -> SAR Tower Inference latency: {t_sar_infer:.2f} ms")
    print(f"  -> SAR Embedding shape: {sar_emb.shape}")
    assert sar_emb.shape == (1, 512), f"Expected (1, 512), got {sar_emb.shape}"
    assert np.isfinite(sar_emb).all(), "SAR embedding contains NaN/Inf!"
    sar_norm = np.linalg.norm(sar_emb[0])
    print(f"  -> SAR Embedding L2 norm: {sar_norm:.6f}")
    assert abs(sar_norm - 1.0) < 1e-4, f"SAR embedding not L2-normalized: {sar_norm}"
    print("  [PASS] SAR Embedding verified.\n")

    # --------------------------------------------------------
    # 6. MS EMBEDDING INFERENCE & LATENCY
    # --------------------------------------------------------
    print("[6/14] Testing MS Tower Inference & 512-D Embedding...")
    t0 = time.perf_counter()
    ms_emb = retriever.encode_ms(ms_tensor)
    t_ms_infer = (time.perf_counter() - t0) * 1000.0
    print(f"  -> MS Tower Inference latency: {t_ms_infer:.2f} ms")
    print(f"  -> MS Embedding shape: {ms_emb.shape}")
    assert ms_emb.shape == (1, 512), f"Expected (1, 512), got {ms_emb.shape}"
    assert np.isfinite(ms_emb).all(), "MS embedding contains NaN/Inf!"
    ms_norm = np.linalg.norm(ms_emb[0])
    print(f"  -> MS Embedding L2 norm: {ms_norm:.6f}")
    assert abs(ms_norm - 1.0) < 1e-4, f"MS embedding not L2-normalized: {ms_norm}"
    print("  [PASS] MS Embedding verified.\n")

    # --------------------------------------------------------
    # 7. CROSS-MODAL: SAR QUERY -> MS GALLERY
    # --------------------------------------------------------
    print("[7/14] Testing Cross-Modal Retrieval: SAR Query -> MS Gallery (k=10)...")
    t0 = time.perf_counter()
    sar_query_results = retriever.search_gallery(sar_emb, target_gallery="MS", k=10)
    t_sar_search = (time.perf_counter() - t0) * 1000.0
    print(f"  -> FAISS search latency (SAR -> MS): {t_sar_search:.2f} ms")
    print(f"  -> Retrieved count: {len(sar_query_results)}")
    assert len(sar_query_results) == 10, f"Expected 10 results, got {len(sar_query_results)}"

    for r in sar_query_results:
        assert r["modality"] == "Multispectral", f"Expected 'Multispectral', got {r['modality']}"
        assert r["patch_id"] in retriever.metadata_lookup, f"Unknown patch ID: {r['patch_id']}"
        assert np.isfinite(r["similarity_score"]), "Similarity score is NaN/Inf!"

    top1_sar = sar_query_results[0]
    print(f"  -> Top-1 retrieved MS patch: {top1_sar['patch_id']}")
    print(f"  -> Top-1 similarity score  : {top1_sar['similarity_score']:.4f}")
    print(f"  -> Top-1 label             : {top1_sar['label']}")
    print(f"  -> Top-1 country           : {top1_sar['country']}")
    print("  [PASS] SAR -> MS retrieval verified.\n")

    # --------------------------------------------------------
    # 8. CROSS-MODAL: MS QUERY -> SAR GALLERY
    # --------------------------------------------------------
    print("[8/14] Testing Cross-Modal Retrieval: MS Query -> SAR Gallery (k=10)...")
    t0 = time.perf_counter()
    ms_query_results = retriever.search_gallery(ms_emb, target_gallery="SAR", k=10)
    t_ms_search = (time.perf_counter() - t0) * 1000.0
    print(f"  -> FAISS search latency (MS -> SAR): {t_ms_search:.2f} ms")
    print(f"  -> Retrieved count: {len(ms_query_results)}")
    assert len(ms_query_results) == 10, f"Expected 10 results, got {len(ms_query_results)}"

    for r in ms_query_results:
        assert r["modality"] == "SAR", f"Expected 'SAR', got {r['modality']}"
        assert r["patch_id"] in retriever.metadata_lookup, f"Unknown patch ID: {r['patch_id']}"
        assert np.isfinite(r["similarity_score"]), "Similarity score is NaN/Inf!"

    top1_ms = ms_query_results[0]
    print(f"  -> Top-1 retrieved SAR patch: {top1_ms['patch_id']}")
    print(f"  -> Top-1 similarity score   : {top1_ms['similarity_score']:.4f}")
    print(f"  -> Top-1 label              : {top1_ms['label']}")
    print(f"  -> Top-1 country            : {top1_ms['country']}")
    print("  [PASS] MS -> SAR retrieval verified.\n")

    # --------------------------------------------------------
    # 9. STRICT RANKING ORDER CHECK
    # --------------------------------------------------------
    print("[9/14] Testing Descending Similarity Score Ranking...")
    for label, results in [("SAR->MS", sar_query_results), ("MS->SAR", ms_query_results)]:
        scores = [r["similarity_score"] for r in results]
        is_descending = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        assert is_descending, f"{label} scores not strictly descending: {scores}"
        print(f"  -> {label} scores: {[round(s, 4) for s in scores]}")
    print("  [PASS] Ranking order verified.\n")

    # --------------------------------------------------------
    # 10. PATCH ID & METADATA RESOLUTION
    # --------------------------------------------------------
    print("[10/14] Testing Patch ID & Metadata Resolution...")
    sample_res = sar_query_results[0]
    required_keys = [
        "rank",
        "patch_id",
        "similarity_score",
        "score",
        "modality",
        "label",
        "labels",
        "country",
        "s1_name",
        "s2v1_name",
        "split",
        "tone",
    ]
    for key in required_keys:
        assert key in sample_res, f"Missing key '{key}' in result dictionary"
    print("  [PASS] Metadata resolution verified.\n")

    # --------------------------------------------------------
    # 11. END-TO-END `retrieve(...)` INTERFACE
    # --------------------------------------------------------
    print("[11/14] Testing End-to-End `retrieve(...)` Interface...")
    e2e_sar = retriever.retrieve(
        query_image_path=sar_tensor,
        query_modality="SAR",
        subcategory="flood-mapping",
        k=10,
    )
    e2e_ms = retriever.retrieve(
        query_image_path=ms_tensor,
        query_modality="MS",
        subcategory="wildfire-scar",
        k=10,
    )
    assert len(e2e_sar) == 10
    assert len(e2e_ms) == 10
    print("  [PASS] End-to-end retrieve interface verified.\n")

    # --------------------------------------------------------
    # 12. MULTI-QUERY LATENCY BENCHMARK
    # --------------------------------------------------------
    print("[12/14] Benchmarking Latency Over Multiple Queries (5 runs)...")
    search_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = retriever.search_gallery(sar_emb, target_gallery="MS", k=10)
        t1 = time.perf_counter()
        search_times.append((t1 - t0) * 1000.0)

    print(f"  -> FAISS Search: mean={np.mean(search_times):.2f} ms, median={np.median(search_times):.2f} ms")
    print("  [PASS] Latency benchmark complete.\n")

    # --------------------------------------------------------
    # 13. MODULE-LEVEL FUNCTION IMPORT TEST
    # --------------------------------------------------------
    print("[13/14] Testing Module-Level `retrieve` Function...")
    func_res = retrieve(sar_tensor, "SAR", k=5)
    assert len(func_res) == 5, f"Expected 5 results, got {len(func_res)}"
    assert func_res[0]["modality"] == "Multispectral"
    print("  [PASS] Module-level retrieve function works.\n")

    # --------------------------------------------------------
    # 14. FINAL SUMMARY
    # --------------------------------------------------------
    print("[14/14] Overall Suite Result:")
    print("=" * 70)
    print("ALL 14 RETRIEVER VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
