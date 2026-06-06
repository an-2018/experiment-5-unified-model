"""Tests for graph_builder.py - Phase 6 Graph MoE Architect."""

import numpy as np
import pytest

from src.data.graph_builder import (
    build_knn_graph,
    build_split_local_graph,
    build_inductive_graph,
    validate_graph_leakage,
    build_multimodal_graph,
)


class TestBuildKNNGraph:
    """Tests for build_knn_graph function."""

    def test_edge_index_shape(self):
        """Verify edge_index has correct shape (2, num_edges)."""
        embeddings = np.random.randn(50, 128)
        edge_index, edge_weight = build_knn_graph(embeddings, k=10)

        assert edge_index.shape[0] == 2, "edge_index should have 2 rows"
        assert edge_index.shape[1] == edge_weight.shape[0], "edges/weights mismatch"

    def test_no_self_edges(self):
        """Verify no self-loops in the graph."""
        embeddings = np.random.randn(50, 128)
        edge_index, _ = build_knn_graph(embeddings, k=10)

        # Check that no edge has src == dst
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        assert np.all(src_nodes != dst_nodes), "Self-edges found"

    def test_symmetric_edges(self):
        """Verify graph is symmetric (or at least each node has k edges)."""
        embeddings = np.random.randn(30, 64)
        k = 5
        edge_index, _ = build_knn_graph(embeddings, k=k)

        n = embeddings.shape[0]
        # Each node should have k outgoing edges (k neighbors)
        unique_srcs, counts = np.unique(edge_index[0], return_counts=True)
        assert np.all(counts == k), f"Expected k={k} edges per node, got {counts}"

    def test_edge_weights_normalized(self):
        """Verify edge weights are in (0, 1] range (similarity scores)."""
        embeddings = np.random.randn(50, 128)
        edge_index, edge_weight = build_knn_graph(embeddings, k=10)

        assert np.all(edge_weight > 0), "Weights should be positive"
        assert np.all(edge_weight <= 1), "Weights should be <= 1 (similarity)"

    def test_k_param_respected(self):
        """Verify k parameter controls edge count."""
        embeddings = np.random.randn(50, 128)
        for k in [3, 5, 10]:
            edge_index, _ = build_knn_graph(embeddings, k=k)
            assert edge_index.shape[1] == 50 * k, f"Expected {50*k} edges for k={k}"


class TestSplitLocalGraph:
    """Tests for build_split_local_graph function."""

    def test_val_edges_no_leakage(self):
        """Verify val edges don't touch train or test nodes."""
        n = 100
        embeddings = np.random.randn(n, 64)
        split_ids = np.array([0] * 60 + [1] * 20 + [2] * 20)  # 60 train, 20 val, 20 test

        graphs, leakage_check = build_split_local_graph(embeddings, split_ids, k=5)

        # Get val node global indices
        val_mask = split_ids == 1
        val_global_indices = np.where(val_mask)[0]
        val_min, val_max = val_global_indices.min(), val_global_indices.max() + 1

        # Check val edges
        val_edge_index = graphs["val"][0]
        val_srcs = val_edge_index[0]  # Local indices within val graph
        val_dsts = val_edge_index[1]

        # In local val graph, indices are [0, val_size). We need to check global.
        # The val graph only contains val nodes, so dsts should be in [0, val_size)
        val_size = 20
        assert np.all(val_dsts < val_size), "Val graph has edges to non-val nodes"

        assert leakage_check["val_leakage_free"], "Val leakage detected"

    def test_test_edges_no_leakage(self):
        """Verify test edges don't touch train or val nodes."""
        n = 100
        embeddings = np.random.randn(n, 64)
        split_ids = np.array([0] * 60 + [1] * 20 + [2] * 20)

        graphs, leakage_check = build_split_local_graph(embeddings, split_ids, k=5)

        # Check test edges stay within test graph
        test_edge_index = graphs["test"][0]
        test_dsts = test_edge_index[1]
        test_size = 20
        assert np.all(test_dsts < test_size), "Test graph has edges to non-test nodes"

        assert leakage_check["test_leakage_free"], "Test leakage detected"


class TestInductiveGraph:
    """Tests for build_inductive_graph function."""

    def test_train_edges_within_train(self):
        """Verify train edges only connect train nodes."""
        train_embeddings = np.random.randn(50, 64)
        test_embeddings = np.random.randn(30, 64)

        train_edge_index, _, _, _ = build_inductive_graph(train_embeddings, test_embeddings, k=5)

        train_max_idx = 50
        train_srcs = train_edge_index[0]
        train_dsts = train_edge_index[1]

        assert np.all(train_srcs < train_max_idx), "Train edge src out of bounds"
        assert np.all(train_dsts < train_max_idx), "Train edge dst out of bounds"

    def test_test_edges_only_to_train(self):
        """Verify test nodes only connect to train nodes (not other test nodes)."""
        train_embeddings = np.random.randn(50, 64)
        test_embeddings = np.random.randn(30, 64)

        _, _, test_edge_index, _ = build_inductive_graph(train_embeddings, test_embeddings, k=5)

        # Test nodes have global indices [50, 80)
        # Test destinations should be in [0, 50) - only train nodes
        test_srcs = test_edge_index[0]
        test_dsts = test_edge_index[1]

        assert np.all(test_srcs >= 50), "Test edge src should be >= 50"
        assert np.all(test_dsts < 50), "Test edge dst should be < 50 (train only)"

    def test_no_test_to_test_edges(self):
        """Verify there are no edges between test nodes."""
        train_embeddings = np.random.randn(50, 64)
        test_embeddings = np.random.randn(30, 64)

        _, _, test_edge_index, _ = build_inductive_graph(train_embeddings, test_embeddings, k=5)

        # Test edges go from test nodes (>=50) to train nodes (<50)
        # There should be NO test->test edges
        test_srcs = test_edge_index[0]
        test_dsts = test_edge_index[1]

        # All destinations should be train nodes (< 50)
        assert np.all(test_dsts < 50), "Found test-to-test edge"


class TestValidateGraphLeakage:
    """Tests for validate_graph_leakage function."""

    def test_val_leakage_free_when_valid(self):
        """Verify val_leakage_free is True when val edges are within bounds."""
        train_size = 60
        val_size = 20

        # Val edges within [60, 80)
        val_edge_index = np.array([
            [60, 61, 65, 70],
            [62, 63, 66, 71]
        ])
        train_edge_index = np.array([[0, 1], [1, 2]])  # dummy
        test_edge_index = np.array([[80, 81], [0, 1]])  # dummy

        result = validate_graph_leakage(train_edge_index, val_edge_index, test_edge_index, train_size, val_size)

        assert result["val_leakage_free"] is True

    def test_val_leakage_detected(self):
        """Verify val_leakage_free is False when val edges touch train nodes."""
        train_size = 60
        val_size = 20

        # Val edge touches train node (node 50) - LEAKAGE
        val_edge_index = np.array([
            [60, 61],
            [50, 63]  # 50 is a train node!
        ])
        train_edge_index = np.array([[0, 1], [1, 2]])
        test_edge_index = np.array([[80, 81], [0, 1]])

        result = validate_graph_leakage(train_edge_index, val_edge_index, test_edge_index, train_size, val_size)

        assert result["val_leakage_free"] is False

    def test_test_leakage_detected(self):
        """Verify test_leakage_free is False when test edges touch train+val boundary."""
        train_size = 60
        val_size = 20

        # Test edge goes to a node >= test_start (80) - LEAKAGE
        # This would be a test-to-test edge which is not allowed in inductive mode
        train_edge_index = np.array([[0, 1], [1, 2]])
        val_edge_index = np.array([[60, 61], [62, 63]])
        test_edge_index = np.array([
            [80, 81],
            [85, 90]  # 85, 90 are test nodes - LEAKAGE!
        ])

        result = validate_graph_leakage(train_edge_index, val_edge_index, test_edge_index, train_size, val_size)

        assert result["test_leakage_free"] is False


class TestMultimodalGraph:
    """Tests for build_multimodal_graph function."""

    def test_cross_dataset_edges_tracked(self):
        """Verify cross-dataset edges have edge_flag=1."""
        embeddings = np.random.randn(100, 64)
        dataset_ids = ["daic"] * 30 + ["mosei"] * 40 + ["fi"] * 30

        edge_index, edge_weight, edge_flags = build_multimodal_graph(
            embeddings, dataset_ids, k=5, cross_dataset_edges=True
        )

        # Check that some edges are cross-dataset
        assert np.any(edge_flags == 1), "No cross-dataset edges found"

    def test_same_dataset_edges_flagged(self):
        """Verify same-dataset edges have edge_flag=0."""
        embeddings = np.random.randn(100, 64)
        dataset_ids = ["daic"] * 30 + ["mosei"] * 40 + ["fi"] * 30

        edge_index, edge_weight, edge_flags = build_multimodal_graph(
            embeddings, dataset_ids, k=5, cross_dataset_edges=True
        )

        # Check that some edges are same-dataset
        assert np.any(edge_flags == 0), "No same-dataset edges found"

    def test_no_cross_edges_when_disabled(self):
        """Verify cross_dataset_edges=False produces all same-dataset edges."""
        embeddings = np.random.randn(100, 64)
        dataset_ids = ["daic"] * 30 + ["mosei"] * 40 + ["fi"] * 30

        _, _, edge_flags = build_multimodal_graph(
            embeddings, dataset_ids, k=5, cross_dataset_edges=False
        )

        assert np.all(edge_flags == 0), "Cross-dataset edges found when disabled"

    def test_edge_count_with_k(self):
        """Verify edge count scales with k parameter."""
        embeddings = np.random.randn(50, 64)
        dataset_ids = ["daic"] * 50

        for k in [3, 5, 10]:
            _, edge_weight, _ = build_multimodal_graph(
                embeddings, dataset_ids, k=k, cross_dataset_edges=False
            )
            expected_edges = 50 * k
            assert len(edge_weight) == expected_edges, f"Expected {expected_edges} edges for k={k}"


def test_validate_graph_no_cross_split_leakage_accepts_clean_graph():
    """Test that validation passes when no cross-split edges exist."""
    from src.data.graph_builder import validate_graph_no_cross_split_leakage

    edge_index = np.array([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    split_ids = np.array([0, 0, 1, 1, 2, 2])

    validate_graph_no_cross_split_leakage(edge_index, split_ids, "test_graph")


def test_validate_graph_no_cross_split_leakage_rejects_cross_split():
    """Test that validation raises ValueError on cross-split edges."""
    from src.data.graph_builder import validate_graph_no_cross_split_leakage

    edge_index = np.array([[0, 3], [3, 0]])  # train node 0 → val node 3 (cross-split!)
    split_ids = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="cross-split edges"):
        validate_graph_no_cross_split_leakage(edge_index, split_ids, "train_graph")


def test_build_multimodal_graph_default_is_safe():
    """Test that build_multimodal_graph defaults to cross_dataset_edges=False."""
    from src.data.graph_builder import build_multimodal_graph
    import inspect

    sig = inspect.signature(build_multimodal_graph)
    default = sig.parameters['cross_dataset_edges'].default
    assert default == False, f"Expected cross_dataset_edges default False, got {default}"


def test_transductive_mode_is_clearly_marked():
    """Test that transductive mode produces expected edge structure with cross-split edges."""
    from src.data.graph_builder import build_multimodal_graph

    embeddings = np.random.randn(6, 8)
    dataset_ids = ["a", "a", "a", "b", "b", "b"]
    split_ids = np.array([0, 0, 1, 1, 2, 2])

    edge_index, _, edge_flags = build_multimodal_graph(embeddings, dataset_ids, k=2, cross_dataset_edges=True)

    src_split = split_ids[edge_index[0]]
    dst_split = split_ids[edge_index[1]]
    cross_split_mask = src_split != dst_split

    assert cross_split_mask.any(), "Transductive mode should produce cross-split edges"