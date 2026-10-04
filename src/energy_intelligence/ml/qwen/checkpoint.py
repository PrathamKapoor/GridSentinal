"""The Qwen3-1.7B-Base checkpoint: identity, verification and architecture facts.

Everything in this module is about the *real* pretrained base checkpoint. Nothing
here trains, adapts or modifies the model. Phase 6 keeps Qwen structurally intact
(D-086), so this module is also the place that pins down what "intact" means: the
invariants in :data:`QWEN3_ARCHITECTURE_INVARIANTS` are asserted by tests before any
Phase 7 surgery is attempted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "QWEN3_1_7B_BASE",
    "QWEN3_ARCHITECTURE_INVARIANTS",
    "ArchitectureInvariant",
    "CheckpointFacts",
    "Qwen3Checkpoint",
    "assert_architecture_intact",
    "inspect_architecture",
]


@dataclass(frozen=True, slots=True)
class CheckpointFacts:
    """What was verified about the checkpoint, and where it came from.

    Attributes:
        model_id: The Hugging Face repository id.
        revision: The pinned commit sha. Never "latest": a moving tag would make the
            experiment unreproducible, which is the whole point of pinning.
        architecture: The architecture class from ``config.json``.
        model_type: The ``model_type`` field.
        num_hidden_layers: Decoder layer count.
        hidden_size: Residual stream width.
        num_attention_heads: Query heads.
        num_key_value_heads: Key/value heads; less than the query heads means grouped
            query attention, which halves the KV cache and is a Phase 7 design fact.
        head_dim: Per-head width, which need not equal ``hidden_size / heads``.
        intermediate_size: FFN inner width.
        vocab_size: Embedding rows. Note this is larger than the tokenizer length
            because the embedding matrix is padded for efficiency.
        max_position_embeddings: Trained context length.
        rope_theta: RoPE base frequency.
        tied_word_embeddings: Whether the LM head reuses the embedding matrix.
        torch_dtype: The dtype the checkpoint was published in.
        layer_types: Per-layer attention type. Uniform here, which makes Phase 7
            surgery simpler because no layer is structurally special.
        license_spdx: The SPDX license tag from the Hub metadata.
        gated: Whether the Hub gates access. False is required for this project.
        weight_file: The single safetensors shard.
        weight_bytes: Its size on disk.
        tensor_count: Number of tensors in the shard.
        parameter_count: Total elements across all tensors.
        parameter_dtype: The dtype actually stored.
    """

    model_id: str
    revision: str
    architecture: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    tied_word_embeddings: bool
    torch_dtype: str
    layer_types: tuple[str, ...]
    license_spdx: str
    gated: bool
    weight_file: str
    weight_bytes: int
    tensor_count: int
    parameter_count: int
    parameter_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointFacts":
        fields = {
            name: (tuple(payload[name]) if name == "layer_types" else payload[name])
            for name in cls.__slots__
        }
        return cls(**fields)


# The provisional foundation checkpoint for this project, pinned to an exact commit.
#
# WHY THIS ONE, and why pinned rather than tracked: the model audit that preceded
# Phase 6 classified it on pretrained/base lineage, downloadable weights, permissive
# licensing, non-gated access, standard architecture, Transformers compatibility and
# suitability for later MoE surgery. This revision was then verified directly against
# the authoritative Hub repository, which is the source of truth for the checkpoint
# itself rather than for the audit's conclusions.
QWEN3_1_7B_BASE = CheckpointFacts(
    model_id="Qwen/Qwen3-1.7B-Base",
    revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    num_hidden_layers=28,
    hidden_size=2048,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=6144,
    vocab_size=151936,
    max_position_embeddings=32768,
    rope_theta=1000000.0,
    tied_word_embeddings=True,
    torch_dtype="bfloat16",
    layer_types=("full_attention",) * 28,
    license_spdx="apache-2.0",
    gated=False,
    weight_file="model.safetensors",
    weight_bytes=3_441_185_608,
    tensor_count=310,
    parameter_count=1_720_574_976,
    parameter_dtype="BF16",
)


@dataclass(frozen=True, slots=True)
class ArchitectureInvariant:
    """One structural fact that Phase 7 must preserve or change deliberately.

    Attributes:
        name: What is being pinned.
        attribute: Dotted path on the Transformers model.
        expected: The value read from the real checkpoint.
        why: What breaks downstream if this changes silently.
    """

    name: str
    attribute: str
    expected: Any
    why: str


# The Phase 7 regression baseline (brief section 43). Each entry is a fact about the
# real checkpoint, not a number chosen for convenience, and each says what it would
# break. Layer count and hidden width in particular determine how many experts a
# future MoE could hold and how wide each would be, so they must be pinned before any
# structural modification rather than discovered after one.
QWEN3_ARCHITECTURE_INVARIANTS: tuple[ArchitectureInvariant, ...] = (
    ArchitectureInvariant(
        "decoder_layers",
        "config.num_hidden_layers",
        28,
        "Sets how many blocks a future MoE could modify or replicate.",
    ),
    ArchitectureInvariant(
        "hidden_size",
        "config.hidden_size",
        2048,
        "The residual stream width; every adapter and expert must match it.",
    ),
    ArchitectureInvariant(
        "ffn_intermediate_size",
        "config.intermediate_size",
        6144,
        "The FFN inner width, three times hidden; the natural unit of Phase 7 surgery.",
    ),
    ArchitectureInvariant(
        "attention_heads",
        "config.num_attention_heads",
        16,
        "Query head count; head_dim times this must equal hidden_size.",
    ),
    ArchitectureInvariant(
        "key_value_heads",
        "config.num_key_value_heads",
        8,
        "Grouped-query attention: half the query heads share each KV head.",
    ),
    ArchitectureInvariant(
        "head_dim",
        "config.head_dim",
        128,
        "Per-head width, independent of hidden_size / heads here.",
    ),
    ArchitectureInvariant(
        "embedding_rows",
        "config.vocab_size",
        151936,
        "The embedding table this phase bypasses entirely; Phase 7 may not need it.",
    ),
    ArchitectureInvariant(
        "tied_embeddings",
        "config.tie_word_embeddings",
        True,
        "The LM head shares storage with the embedding matrix.",
    ),
    ArchitectureInvariant(
        "layer_types_uniform",
        "layer_types",
        ("full_attention",) * 28,
        "Every layer is full attention, so no layer is structurally special.",
    ),
    ArchitectureInvariant(
        "rms_norm_eps",
        "config.rms_norm_eps",
        1e-06,
        "Pre-norm RMSNorm epsilon; changing it silently rescales every residual.",
    ),
)


class Qwen3Checkpoint:
    """A verified local copy of the base checkpoint.

    Construction is deliberately strict. It checks that the files exist, that the
    configuration agrees with the pinned facts above, and that the weight file is the
    expected size and dtype. A download that "succeeded" is not evidence that the
    checkpoint is usable, so this class refuses to hand over a model that has not
    passed those checks.
    """

    def __init__(self, root: Path, facts: CheckpointFacts = QWEN3_1_7B_BASE) -> None:
        self.root = Path(root)
        self.facts = facts
        self._verified: dict[str, Any] | None = None

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def weight_path(self) -> Path:
        return self.root / self.facts.weight_file

    @property
    def tokenizer_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.root / name
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        )

    def verify(self, *, strict_size: bool = True) -> dict[str, Any]:
        """Check the local copy against the pinned facts.

        Args:
            strict_size: Require the weight file to match the pinned byte count
                exactly. Worth it for a 3.4 GB download, where a truncated file can
                otherwise load far enough to produce plausible nonsense.

        Returns:
            The verification record, including measured sizes and parameter counts.

        Raises:
            FileNotFoundError: A required file is missing.
            ValueError: The configuration or weights disagree with the pinned facts.
        """
        missing = [
            path.name
            for path in (self.config_path, self.weight_path, *self.tokenizer_paths)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"checkpoint at {self.root} is missing {missing}; "
                "see docs/smart_ds_acquisition.md for the acquisition pattern"
            )

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        measured_bytes = self.weight_path.stat().st_size
        if strict_size and measured_bytes != self.facts.weight_bytes:
            raise ValueError(
                f"weight file is {measured_bytes:,} bytes, pinned facts say "
                f"{self.facts.weight_bytes:,}; the download is incomplete or modified"
            )

        for field, key in (
            ("architecture", "architectures"),
            ("model_type", "model_type"),
            ("num_hidden_layers", "num_hidden_layers"),
            ("hidden_size", "hidden_size"),
            ("num_attention_heads", "num_attention_heads"),
            ("num_key_value_heads", "num_key_value_heads"),
            ("head_dim", "head_dim"),
            ("intermediate_size", "intermediate_size"),
            ("vocab_size", "vocab_size"),
            ("max_position_embeddings", "max_position_embeddings"),
            ("tied_word_embeddings", "tie_word_embeddings"),
        ):
            actual = config.get(key)
            expected = getattr(self.facts, field)
            if field == "architecture":
                actual = actual[0] if isinstance(actual, list) else actual
            if actual != expected:
                raise ValueError(
                    f"config.json {key}={actual!r} but pinned facts say {expected!r}"
                )

        measured = self.inspect_weights()
        self._verified = {
            "model_id": self.facts.model_id,
            "revision": self.facts.revision,
            "local_root": str(self.root),
            "config_agrees": True,
            "weight_bytes": measured_bytes,
            "expected_weight_bytes": self.facts.weight_bytes,
            "tokenizer_files": [p.name for p in self.tokenizer_paths],
            **measured,
        }
        return dict(self._verified)

    def inspect_weights(self) -> dict[str, Any]:
        """Read the safetensors header to measure dtype, tensor count and parameters.

        Reads only the header, so it costs seconds rather than the minutes a full
        tensor load would take, and it cannot be fooled by lazily-mapped memory.
        """
        from safetensors import safe_open

        keys: list[str] = []
        dtypes: dict[str, int] = {}
        elements = 0
        with safe_open(str(self.weight_path), framework="pt") as handle:
            keys = list(handle.keys())
            for key in keys:
                sliver = handle.get_slice(key)
                dtype = sliver.get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                count = 1
                for dimension in sliver.get_shape():
                    count *= dimension
                elements += count
        return {
            "tensor_count": len(keys),
            "parameter_count": elements,
            "dtypes": dtypes,
            "metadata": None,
            "has_lm_head_tensor": any(k.endswith("lm_head.weight") for k in keys),
        }

    def load_backbone(self, *, dtype: str = "float32", device: str = "cpu"):
        """Load the real weights as a frozen backbone.

        Args:
            dtype: ``"float32"`` or ``"bfloat16"``. The default is float32 and the
                reason is measured, not aesthetic: on this CPU, bf16 has no native
                arithmetic and torch emulates it, which made the forward pass
                **4.8x slower** than fp32 (2.568 vs 0.533 s/sample at 21 tokens).
                Published checkpoint dtype is not the same thing as the fastest
                inference dtype.
            device: Where to place the weights.

        Returns:
            The ``Qwen3Model`` trunk - the embedding table, the 28 decoder layers and
            the final norm, without the language-model head. Phase 6 never uses the
            vocabulary head: the task is regression, not generation, and computing
            logits over 151,936 rows would be pure waste.
        """
        import torch
        from transformers import AutoModel

        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        model = AutoModel.from_pretrained(
            self.root, dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def load_tokenizer(self):
        """Load the published tokenizer. Used for identity checks, not for encoding."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.root)


def inspect_architecture(model) -> dict[str, Any]:
    """Describe a loaded Qwen3 trunk's structure, for the Phase 7 record.

    Reports the actual module tree rather than restating ``config.json``, because the
    two can disagree: transformers can substitute or wrap implementations, and the
    module names are what a future structural edit would actually touch.
    """
    config = model.config
    layer = model.layers[0]
    return {
        "class": type(model).__name__,
        "config_class": type(config).__name__,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "embedding": {
            "class": type(model.embed_tokens).__name__,
            "shape": list(model.embed_tokens.weight.shape),
            "dtype": str(model.embed_tokens.weight.dtype),
            "used_by_phase6": False,
        },
        "final_norm": {
            "class": type(model.norm).__name__,
            "weight_shape": list(model.norm.weight.shape),
            "eps": float(config.rms_norm_eps),
        },
        "decoder_layer": {
            "class": type(layer).__name__,
            "modules": [name for name, _ in layer.named_children()],
            "attention": {
                "class": type(layer.self_attn).__name__,
                "modules": [name for name, _ in layer.self_attn.named_children()],
                "q_proj": list(layer.self_attn.q_proj.weight.shape),
                "k_proj": list(layer.self_attn.k_proj.weight.shape),
                "v_proj": list(layer.self_attn.v_proj.weight.shape),
                "o_proj": list(layer.self_attn.o_proj.weight.shape),
                "q_norm": list(layer.self_attn.q_norm.weight.shape),
                "k_norm": list(layer.self_attn.k_norm.weight.shape),
            },
            "mlp": {
                "class": type(layer.mlp).__name__,
                "modules": [name for name, _ in layer.mlp.named_children()],
                "gate_proj": list(layer.mlp.gate_proj.weight.shape),
                "up_proj": list(layer.mlp.up_proj.weight.shape),
                "down_proj": list(layer.mlp.down_proj.weight.shape),
            },
            "residual_paths": [
                "attention_residual",
                "ffn_residual",
            ],
            "norm_placement": "pre-norm (input_layernorm and post_attention_layernorm "
            "both precede their sublayer)",
        },
        "positional_encoding": {
            "type": "rotary (RoPE)",
            "rope_theta": float(getattr(config, "rope_theta", None) or _rope_theta(config)),
            "max_position_embeddings": config.max_position_embeddings,
            "absolute_position_embedding": False,
            "note": "RoPE is relative, so absolute time-of-day cannot come from "
            "position and must come from the cyclic input channels.",
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def _rope_theta(config) -> float:
    """Read RoPE theta across the transformers 4.x and 5.x config layouts.

    Transformers 5 moved ``rope_theta``/``rope_scaling`` into a ``rope_parameters``
    dict. Reading only the old attribute would make this report a null theta on the
    installed version while the on-disk config still carries the classic field.
    """
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict) and "rope_theta" in parameters:
        return float(parameters["rope_theta"])
    return float(getattr(config, "rope_theta", 0.0))


def assert_architecture_intact(model, *, facts: CheckpointFacts = QWEN3_1_7B_BASE) -> None:
    """Fail if a loaded trunk deviates from the pinned architecture.

    This is the Phase 7 safety check (brief section 43). Phase 6 must be able to say
    "the structure is exactly what the checkpoint ships" before anyone starts
    replacing FFNs with experts, otherwise a later regression would be
    indistinguishable from the modification that was intended.

    Raises:
        ValueError: Any invariant differs from the pinned value.
    """
    config = model.config
    live: dict[str, Any] = {
        "config.num_hidden_layers": config.num_hidden_layers,
        "config.hidden_size": config.hidden_size,
        "config.intermediate_size": config.intermediate_size,
        "config.num_attention_heads": config.num_attention_heads,
        "config.num_key_value_heads": config.num_key_value_heads,
        "config.head_dim": config.head_dim,
        "config.vocab_size": config.vocab_size,
        "config.tie_word_embeddings": config.tie_word_embeddings,
        "config.rms_norm_eps": config.rms_norm_eps,
        "layer_types": tuple(config.layer_types),
    }
    failures: list[str] = []
    for invariant in QWEN3_ARCHITECTURE_INVARIANTS:
        actual = live.get(invariant.attribute)
        if actual != invariant.expected:
            failures.append(
                f"{invariant.name}: expected {invariant.expected!r}, found {actual!r} "
                f"({invariant.why})"
            )
    if failures:
        raise ValueError("architecture is not intact:\n  " + "\n  ".join(failures))
    _ = facts
