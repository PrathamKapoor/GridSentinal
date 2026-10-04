"""Tests for the Phase 6 checkpoint contract and the Phase 7 surgery baseline.

These tests exist for two different audiences. Some of them check that Phase 6 did not
corrupt its own inputs - the pinned revision, the byte count, the token count. The
rest exist for **Phase 7**, which will start replacing modules inside this model, and
which needs a regression baseline established *before* anyone changes anything
(brief section 43).

Tests that need the 3.44 GB checkpoint are skipped when it is absent, so the suite
still runs on a machine that has not downloaded it. The skip is explicit and loud
rather than silent, because "the architecture tests did not run" and "the architecture
tests passed" must never look the same in a CI log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_intelligence.ml.qwen.checkpoint import (
    QWEN3_1_7B_BASE,
    QWEN3_ARCHITECTURE_INVARIANTS,
    Qwen3Checkpoint,
    assert_architecture_intact,
    inspect_architecture,
)

CHECKPOINT_ROOT = Path("models/qwen3-1.7b-base")
requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT_ROOT / "model.safetensors").is_file(),
    reason="Qwen3-1.7B-Base is not downloaded; run the Phase 6 acquisition first",
)


# ======================================================================
# The pinned identity. Cheap, and the guard against silently moving target.
# ======================================================================


def test_the_foundation_checkpoint_is_pinned_to_an_exact_revision() -> None:
    """A moving tag would make the experiment irreproducible."""
    facts = QWEN3_1_7B_BASE
    assert facts.model_id == "Qwen/Qwen3-1.7B-Base"
    assert facts.revision == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    assert len(facts.revision) == 40
    assert not facts.revision.endswith(("main", "latest"))
    int(facts.revision, 16)  # it is a sha, not a label


def test_the_checkpoint_is_the_base_model_not_an_instruct_variant() -> None:
    """Phase 6 required the pretrained base, never instruct, chat or a fine-tune."""
    model_id = QWEN3_1_7B_BASE.model_id.lower()
    assert "base" in model_id
    for forbidden in ("instruct", "chat", "gguf", "awq", "gptq", "lora", "adapter"):
        assert forbidden not in model_id


def test_licensing_and_access_match_the_verified_audit() -> None:
    """Recorded, not interpreted: these are the values read from the Hub."""
    assert QWEN3_1_7B_BASE.license_spdx == "apache-2.0"
    assert QWEN3_1_7B_BASE.gated is False


def test_the_parameter_count_is_near_the_nominal_1_7b() -> None:
    """The checkpoint is described as ~1.7B, so the claim must be checkable."""
    facts = QWEN3_1_7B_BASE
    assert 1_600_000_000 < facts.parameter_count < 1_800_000_000


def test_the_embedding_is_tied_so_no_lm_head_tensor_is_stored() -> None:
    """A stored lm_head would double-count the embedding in any parameter total."""
    assert QWEN3_1_7B_BASE.tied_word_embeddings is True


# ======================================================================
# Architecture invariants: the Phase 7 regression baseline
# ======================================================================


def test_every_layer_is_full_attention_so_no_layer_is_special() -> None:
    """Phase 7 surgery is simpler when layer types are uniform, so it is pinned."""
    facts = QWEN3_1_7B_BASE
    assert len(facts.layer_types) == facts.num_hidden_layers
    assert set(facts.layer_types) == {"full_attention"}


def test_grouped_query_attention_is_half_kv_heads() -> None:
    """GQA halves the KV cache; a Phase 7 expert split would have to respect it."""
    facts = QWEN3_1_7B_BASE
    assert facts.num_key_value_heads * 2 == facts.num_attention_heads


def test_attention_head_geometry_is_consistent() -> None:
    facts = QWEN3_1_7B_BASE
    assert facts.num_attention_heads * facts.head_dim == facts.hidden_size
    assert facts.num_key_value_heads * facts.head_dim == 1024


def test_the_ffn_is_three_times_the_residual_width() -> None:
    """Qwen3's 3x FFN; the natural unit of a Phase 7 expert swap."""
    assert QWEN3_1_7B_BASE.intermediate_size == 3 * QWEN3_1_7B_BASE.hidden_size


def test_the_invariant_table_has_no_duplicate_attributes() -> None:
    attributes = [i.attribute for i in QWEN3_ARCHITECTURE_INVARIANTS]
    assert len(attributes) == len(set(attributes))


def test_every_invariant_records_why_it_matters() -> None:
    """An invariant without a stated consequence is documentation, not a check."""
    for invariant in QWEN3_ARCHITECTURE_INVARIANTS:
        assert invariant.why.strip()
        assert invariant.name.strip()


def test_the_pinned_values_match_the_dataclass() -> None:
    """The invariant table and the facts object must not drift apart."""
    live = {
        "config.num_hidden_layers": QWEN3_1_7B_BASE.num_hidden_layers,
        "config.hidden_size": QWEN3_1_7B_BASE.hidden_size,
        "config.intermediate_size": QWEN3_1_7B_BASE.intermediate_size,
        "config.num_attention_heads": QWEN3_1_7B_BASE.num_attention_heads,
        "config.num_key_value_heads": QWEN3_1_7B_BASE.num_key_value_heads,
        "config.head_dim": QWEN3_1_7B_BASE.head_dim,
        "config.vocab_size": QWEN3_1_7B_BASE.vocab_size,
        "config.tie_word_embeddings": QWEN3_1_7B_BASE.tied_word_embeddings,
        "config.rms_norm_eps": 1e-06,
        "layer_types": QWEN3_1_7B_BASE.layer_types,
    }
    for invariant in QWEN3_ARCHITECTURE_INVARIANTS:
        assert live[invariant.attribute] == invariant.expected, invariant.name


def test_assert_architecture_intact_rejects_a_modified_width() -> None:
    """The guard must actually fail, or it is decoration."""

    class FakeConfig:
        num_hidden_layers = 28
        hidden_size = 2048
        intermediate_size = 6144
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128
        vocab_size = 151936
        tie_word_embeddings = True
        rms_norm_eps = 1e-06
        layer_types = ["full_attention"] * 28

    class FakeModel:
        config = FakeConfig()

    assert_architecture_intact(FakeModel())  # intact: passes

    FakeConfig.hidden_size = 4096  # a Phase 7 modification
    with pytest.raises(ValueError, match="hidden_size"):
        assert_architecture_intact(FakeModel())


def test_assert_architecture_intact_rejects_a_changed_layer_count() -> None:
    class FakeConfig:
        num_hidden_layers = 14  # layers deleted
        hidden_size = 2048
        intermediate_size = 6144
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128
        vocab_size = 151936
        tie_word_embeddings = True
        rms_norm_eps = 1e-06
        layer_types = ["full_attention"] * 14

    class FakeModel:
        config = FakeConfig()

    with pytest.raises(ValueError, match="decoder_layers|layer_types"):
        assert_architecture_intact(FakeModel())


# ======================================================================
# Verification against the real files
# ======================================================================


def test_verification_fails_loudly_when_files_are_missing(tmp_path: Path) -> None:
    """A partial download must not look like a usable checkpoint."""
    with pytest.raises(FileNotFoundError, match="missing"):
        Qwen3Checkpoint(tmp_path).verify()


def test_verification_rejects_a_truncated_weight_file(tmp_path: Path) -> None:
    """3.44 GB can download partially; a short file must not pass verification."""
    (tmp_path / "model.safetensors").write_bytes(b"not a checkpoint")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 28,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 6144,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "tied_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bytes"):
        Qwen3Checkpoint(tmp_path).verify()


def test_verification_rejects_a_config_that_disagrees_with_the_pinned_facts(
    tmp_path: Path,
) -> None:
    """If config.json says something else, the pinned facts are stale."""
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 30,  # not the checkpoint we pinned
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 6144,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "tied_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_hidden_layers"):
        Qwen3Checkpoint(tmp_path, facts=QWEN3_1_7B_BASE).verify(
            strict_size=False
        )


@requires_checkpoint
def test_the_real_checkpoint_verifies_and_agrees_with_the_pinned_facts() -> None:
    record = Qwen3Checkpoint(CHECKPOINT_ROOT).verify()
    assert record["config_agrees"] is True
    assert record["weight_bytes"] == QWEN3_1_7B_BASE.weight_bytes
    assert record["tensor_count"] == QWEN3_1_7B_BASE.tensor_count
    assert record["parameter_count"] == QWEN3_1_7B_BASE.parameter_count
    assert record["dtypes"] == {QWEN3_1_7B_BASE.parameter_dtype: QWEN3_1_7B_BASE.tensor_count}
    # Tied embeddings: the head must NOT be a second stored tensor.
    assert record["has_lm_head_tensor"] is False


@requires_checkpoint
def test_the_real_backbone_loads_with_every_parameter_frozen() -> None:
    """Phase 6's configuration is a frozen backbone; this proves it is actually frozen."""
    checkpoint = Qwen3Checkpoint(CHECKPOINT_ROOT)
    backbone = checkpoint.load_backbone()
    assert all(not p.requires_grad for p in backbone.parameters())
    total = sum(p.numel() for p in backbone.parameters())
    assert total == QWEN3_1_7B_BASE.parameter_count
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    assert trainable == 0


@requires_checkpoint
def test_the_real_backbone_matches_every_architecture_invariant() -> None:
    backbone = Qwen3Checkpoint(CHECKPOINT_ROOT).load_backbone()
    assert_architecture_intact(backbone)


@requires_checkpoint
def test_inspecting_the_real_architecture_finds_the_phase7_surgery_sites() -> None:
    """Phase 7 needs to know the module names it will touch, read from the model."""
    backbone = Qwen3Checkpoint(CHECKPOINT_ROOT).load_backbone()
    report = inspect_architecture(backbone)
    layer = report["decoder_layer"]
    assert layer["class"] == "Qwen3DecoderLayer"
    assert {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"} <= set(
        layer["modules"]
    )
    assert layer["attention"]["q_proj"] == [2048, 2048]
    assert layer["attention"]["k_proj"] == [1024, 2048]  # GQA: half the query width
    assert layer["mlp"]["gate_proj"] == [6144, 2048]
    assert report["embedding"]["shape"] == [151936, 2048]
    assert report["embedding"]["used_by_phase6"] is False
    assert report["positional_encoding"]["type"] == "rotary (RoPE)"
    assert report["positional_encoding"]["absolute_position_embedding"] is False
    assert report["final_norm"]["eps"] == pytest.approx(1e-06)
    assert report["parameter_count"] == QWEN3_1_7B_BASE.parameter_count
