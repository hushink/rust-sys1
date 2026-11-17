#!/usr/bin/env python3
"""Export a Laya safetensors checkpoint to the sys1 ONNX scoring contract.

This is a pure NumPy/ONNX translator: it does not need PyTorch or the Laya
Python package. The exported graph implements the same ModernBERT encoder,
type embedding, decision head, marker gather, and scorer used by
`sys1-laya`'s candle backend.

Inputs:

    input_ids      int64 [batch, seq]
    attention_mask int64 [batch, seq]
    type_ids       int64 [batch]
    markers        int64 [n]       flattened row * seq + position

Output:

    logits         float32 [n]

Usage:

    uv run --with onnx --with safetensors --with numpy --python 3.13 \
      scripts/export_laya_onnx.py /path/to/laya-multilingual /path/to/laya-onnx

The output directory receives `model.onnx`, `onnx.json`, `rl_agent_config.json`,
and the tokenizer files, so it can be served directly by `sys1 serve`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

ATTENTION_MASK_VALUE = np.float32(-10000.0)
HEAD_MASK_VALUE = np.float32(-1_000_000_000.0)
HEAD_NORM_EPS = 1e-5


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counter = 0
        self._names: set[str] = set()

    def name(self, hint: str) -> str:
        base = hint.replace(".", "_").replace("/", "_")
        while True:
            candidate = f"{base}_{self._counter}"
            self._counter += 1
            if candidate not in self._names:
                self._names.add(candidate)
                return candidate

    def init(self, array: np.ndarray, hint: str = "const") -> str:
        name = self.name(hint)
        array = np.asarray(array)
        if array.ndim > 0:
            array = np.ascontiguousarray(array)
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def f32(self, value: float, hint: str = "const") -> str:
        return self.init(np.array(value, dtype=np.float32), hint)

    def i64(self, value: int, hint: str = "const") -> str:
        return self.init(np.array(value, dtype=np.int64), hint)

    def i64s(self, values: list[int], hint: str = "const") -> str:
        return self.init(np.array(values, dtype=np.int64), hint)

    def op(
        self,
        op_type: str,
        inputs: list[str],
        outputs: int = 1,
        hint: str | None = None,
        **attrs,
    ) -> str | list[str]:
        names = [self.name(hint or op_type.lower()) for _ in range(outputs)]
        self.nodes.append(
            helper.make_node(op_type, inputs, names, name=names[0], **attrs)
        )
        return names[0] if outputs == 1 else names

    def alias(self, x: str, name: str) -> str:
        self.nodes.append(helper.make_node("Identity", [x], [name], name=name))
        return name

    def unary(self, op_type: str, x: str) -> str:
        return self.op(op_type, [x])

    def binary(self, op_type: str, left: str, right: str) -> str:
        return self.op(op_type, [left, right])

    def matmul(self, left: str, right: str) -> str:
        return self.binary("MatMul", left, right)

    def reshape(self, x: str, shape: list[int], hint: str = "reshape") -> str:
        return self.op("Reshape", [x, self.i64s(shape, "shape")], hint=hint)

    def transpose(self, x: str, perm: list[int], hint: str = "transpose") -> str:
        return self.op("Transpose", [x], perm=perm, hint=hint)

    def gather(self, x: str, indices: str, axis: int = 0, hint: str = "gather") -> str:
        return self.op("Gather", [x, indices], axis=axis, hint=hint)

    def slice(self, x: str, start: int, end: int, axis: int) -> str:
        return self.op(
            "Slice",
            [
                x,
                self.i64s([start], "slice_start"),
                self.i64s([end], "slice_end"),
                self.i64s([axis], "slice_axis"),
            ],
            hint="slice",
        )

    def linear(
        self,
        x: str,
        weight: np.ndarray,
        hint: str,
        bias: np.ndarray | None = None,
    ) -> str:
        w = self.init(weight.T.astype(np.float32), f"w_{hint}")
        y = self.matmul(x, w)
        if bias is not None:
            b = self.init(bias.astype(np.float32), f"b_{hint}")
            y = self.binary("Add", y, b)
        return y

    def layer_norm(
        self,
        x: str,
        weight: np.ndarray,
        hint: str,
        bias: np.ndarray | None = None,
        eps: float = 1e-5,
    ) -> str:
        inputs = [x, self.init(weight.astype(np.float32), f"ln_w_{hint}")]
        if bias is not None:
            inputs.append(self.init(bias.astype(np.float32), f"ln_b_{hint}"))
        return self.op(
            "LayerNormalization",
            inputs,
            axis=-1,
            epsilon=float(eps),
            hint=f"ln_{hint}",
        )

    def gelu_erf(self, x: str) -> str:
        inv_sqrt2 = self.f32(float(1.0 / np.sqrt(2.0)), "inv_sqrt2")
        one = self.f32(1.0, "one")
        half = self.f32(0.5, "half")
        scaled = self.binary("Mul", x, inv_sqrt2)
        erf = self.unary("Erf", scaled)
        shifted = self.binary("Add", erf, one)
        scaled = self.binary("Mul", shifted, half)
        return self.binary("Mul", scaled, x)

    def model(self, inputs: list[onnx.ValueInfoProto], output: str) -> onnx.ModelProto:
        graph = helper.make_graph(
            self.nodes,
            "laya_onnx",
            inputs,
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, ["n"])],
            self.initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            producer_name="sys1-laya-exporter",
        )
        model.ir_version = 8
        onnx.checker.check_model(model, full_check=True)
        return model


class Weights:
    def __init__(self, path: Path) -> None:
        self._file = safe_open(path, framework="numpy")
        self.keys = set(self._file.keys())

    def has(self, key: str) -> bool:
        return key in self.keys

    def get(self, key: str) -> np.ndarray:
        return np.asarray(self._file.get_tensor(key), dtype=np.float32)


def rope_tables(
    max_positions: int, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = np.array(
        [
            1.0 / np.float32(theta ** (index / head_dim))
            for index in range(0, head_dim, 2)
        ],
        dtype=np.float32,
    )
    positions = np.arange(max_positions, dtype=np.float32).reshape(-1, 1)
    frequencies = (positions @ inv_freq.reshape(1, -1)).astype(np.float32)
    return np.cos(frequencies, dtype=np.float32), np.sin(frequencies, dtype=np.float32)


def rope(
    b: Builder,
    x: str,
    cos4: str,
    sin4: str,
    head_dim: int,
) -> str:
    first = b.slice(x, 0, head_dim // 2, -1)
    second = b.slice(x, head_dim // 2, head_dim, -1)
    rotated = b.op(
        "Concat",
        [b.unary("Neg", second), first],
        axis=-1,
        hint="rotate_half",
    )
    return b.binary(
        "Add",
        b.binary("Mul", x, cos4),
        b.binary("Mul", rotated, sin4),
    )


def attention_mask(b: Builder, attention_mask: str) -> dict[str, str]:
    mask = b.op("Cast", [attention_mask], to=TensorProto.FLOAT, hint="mask_f32")
    one = b.f32(1.0, "mask_one")
    padding = b.binary(
        "Mul",
        b.binary("Sub", one, mask),
        b.f32(float(ATTENTION_MASK_VALUE), "mask_value"),
    )
    padding4 = b.op(
        "Unsqueeze",
        [padding, b.i64s([1, 2], "mask_axes")],
        hint="padding4",
    )
    head_padding = b.binary(
        "Mul",
        b.binary("Sub", one, mask),
        b.f32(float(HEAD_MASK_VALUE), "head_mask_value"),
    )
    head_mask4 = b.op(
        "Unsqueeze",
        [head_padding, b.i64s([1, 2], "head_mask_axes")],
        hint="head_mask4",
    )

    shape = b.op("Shape", [attention_mask], hint="input_shape")
    seq = b.gather(shape, b.i64(1, "seq_index"), hint="seq")
    positions = b.op(
        "Range",
        [b.i64(0, "pos_start"), seq, b.i64(1, "pos_delta")],
        hint="positions",
    )
    row = b.op(
        "Unsqueeze",
        [positions, b.i64s([1], "row_axes")],
        hint="row_positions",
    )
    column = b.op(
        "Unsqueeze",
        [positions, b.i64s([0], "col_axes")],
        hint="col_positions",
    )
    distance = b.op(
        "Abs",
        [b.binary("Sub", row, column)],
        hint="position_distance",
    )
    return {
        "padding4": padding4,
        "head_mask4": head_mask4,
        "distance": distance,
        "positions": positions,
        "seq": seq,
    }


def local_mask(
    b: Builder,
    distance: str,
    padding4: str,
    window: int,
) -> str:
    outside = b.op(
        "Greater",
        [distance, b.i64(window, "local_window")],
        hint="outside_window",
    )
    mask = b.op(
        "Where",
        [
            outside,
            b.f32(float(ATTENTION_MASK_VALUE), "local_mask_value"),
            b.f32(0.0, "local_mask_zero"),
        ],
        hint="local_mask",
    )
    mask4 = b.op(
        "Unsqueeze",
        [mask, b.i64s([0, 1], "local_mask_axes")],
        hint="local_mask4",
    )
    return b.binary("Add", padding4, mask4)


def rope_terms(
    b: Builder,
    positions: str,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    head_dim: int,
    hint: str,
) -> tuple[str, str]:
    cos_name = b.init(cos_table, f"cos_table_{hint}")
    sin_name = b.init(sin_table, f"sin_table_{hint}")
    cos_pos = b.gather(cos_name, positions, hint=f"cos_pos_{hint}")
    sin_pos = b.gather(sin_name, positions, hint=f"sin_pos_{hint}")
    cos_dup = b.op("Concat", [cos_pos, cos_pos], axis=1, hint=f"cos_dup_{hint}")
    sin_dup = b.op("Concat", [sin_pos, sin_pos], axis=1, hint=f"sin_dup_{hint}")
    cos4 = b.reshape(cos_dup, [1, 1, -1, head_dim], f"cos4_{hint}")
    sin4 = b.reshape(sin_dup, [1, 1, -1, head_dim], f"sin4_{hint}")
    return cos4, sin4


def encoder_attention(
    b: Builder,
    x: str,
    qkv_weight: np.ndarray,
    out_weight: np.ndarray,
    heads: int,
    head_dim: int,
    mask: str,
    cos4: str,
    sin4: str,
    hint: str,
) -> str:
    hidden = heads * head_dim
    qkv = b.linear(x, qkv_weight, f"{hint}.qkv")
    qkv = b.reshape(qkv, [0, 0, 3, heads, head_dim], f"{hint}.qkv_reshape")
    qkv = b.transpose(qkv, [2, 0, 3, 1, 4], f"{hint}.qkv_transpose")
    q = b.gather(qkv, b.i64(0, "q_index"), hint=f"{hint}.q")
    k = b.gather(qkv, b.i64(1, "k_index"), hint=f"{hint}.k")
    v = b.gather(qkv, b.i64(2, "v_index"), hint=f"{hint}.v")

    q = rope(b, q, cos4, sin4, head_dim)
    k = rope(b, k, cos4, sin4, head_dim)

    scores = b.matmul(q, b.transpose(k, [0, 1, 3, 2], f"{hint}.k_t"))
    scores = b.binary(
        "Mul",
        scores,
        b.f32(float(head_dim**-0.5), "attention_scale"),
    )
    scores = b.binary("Add", scores, mask)
    probabilities = b.op("Softmax", [scores], axis=-1, hint=f"{hint}.softmax")
    context = b.matmul(probabilities, v)
    context = b.transpose(context, [0, 2, 1, 3], f"{hint}.context_transpose")
    context = b.reshape(context, [0, 0, hidden], f"{hint}.context_reshape")
    return b.linear(context, out_weight, f"{hint}.out")


def head_attention(
    b: Builder,
    x: str,
    weights: Weights,
    hidden: int,
    heads: int,
    head_dim: int,
    mask: str,
    index: int,
) -> str:
    hint = f"head_{index}"
    norm1 = b.layer_norm(
        x,
        weights.get(f"head.layers.{index}.norm1.weight"),
        f"{hint}.norm1",
        weights.get(f"head.layers.{index}.norm1.bias"),
        HEAD_NORM_EPS,
    )
    qkv = b.linear(
        norm1,
        weights.get(f"head.layers.{index}.self_attn.in_proj_weight"),
        f"{hint}.qkv",
        weights.get(f"head.layers.{index}.self_attn.in_proj_bias"),
    )
    qkv = b.reshape(qkv, [0, 0, 3, heads, head_dim], f"{hint}.qkv_reshape")
    qkv = b.transpose(qkv, [2, 0, 3, 1, 4], f"{hint}.qkv_transpose")
    q = b.gather(qkv, b.i64(0, "q_index"), hint=f"{hint}.q")
    k = b.gather(qkv, b.i64(1, "k_index"), hint=f"{hint}.k")
    v = b.gather(qkv, b.i64(2, "v_index"), hint=f"{hint}.v")

    scores = b.matmul(q, b.transpose(k, [0, 1, 3, 2], f"{hint}.k_t"))
    scores = b.binary(
        "Mul",
        scores,
        b.f32(float(head_dim**-0.5), "head_scale"),
    )
    scores = b.binary("Add", scores, mask)
    probabilities = b.op("Softmax", [scores], axis=-1, hint=f"{hint}.softmax")
    context = b.matmul(probabilities, v)
    context = b.transpose(context, [0, 2, 1, 3], f"{hint}.context_transpose")
    context = b.reshape(context, [0, 0, hidden], f"{hint}.context_reshape")
    attention = b.linear(
        context,
        weights.get(f"head.layers.{index}.self_attn.out_proj.weight"),
        f"{hint}.out",
        weights.get(f"head.layers.{index}.self_attn.out_proj.bias"),
    )
    x = b.binary("Add", x, attention)

    feed_forward_input = b.layer_norm(
        x,
        weights.get(f"head.layers.{index}.norm2.weight"),
        f"{hint}.norm2",
        weights.get(f"head.layers.{index}.norm2.bias"),
        HEAD_NORM_EPS,
    )
    feed_forward = b.linear(
        feed_forward_input,
        weights.get(f"head.layers.{index}.linear1.weight"),
        f"{hint}.linear1",
        weights.get(f"head.layers.{index}.linear1.bias"),
    )
    feed_forward = b.unary("Relu", feed_forward)
    feed_forward = b.linear(
        feed_forward,
        weights.get(f"head.layers.{index}.linear2.weight"),
        f"{hint}.linear2",
        weights.get(f"head.layers.{index}.linear2.bias"),
    )
    return b.binary("Add", x, feed_forward)


def build_model(checkpoint: Path) -> onnx.ModelProto:
    config = json.loads((checkpoint / "encoder" / "config.json").read_text())
    agent = json.loads((checkpoint / "rl_agent_config.json").read_text())
    weights = Weights(checkpoint / "model.safetensors")

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = hidden // heads
    layers = int(config["num_hidden_layers"])
    intermediate = int(config["intermediate_size"])
    eps = float(config.get("layer_norm_eps", 1e-5))
    global_every = int(config["global_attn_every_n_layers"])
    local_window = int(config["local_attention"]) // 2
    max_positions = int(config["max_position_embeddings"])
    rope_global = float(
        config.get("rope_parameters", {})
        .get("full_attention", {})
        .get("rope_theta", 10000.0)
    )
    rope_local = float(
        config.get("rope_parameters", {})
        .get("sliding_attention", {})
        .get("rope_theta", rope_global)
    )
    head_layers = int(agent["head_layers"])
    head_heads = hidden // 64
    if head_heads == 0:
        raise ValueError(f"invalid decision-head geometry for hidden={hidden}")
    head_dim_h = hidden // head_heads

    cos_global, sin_global = rope_tables(max_positions, head_dim, rope_global)
    cos_local, sin_local = rope_tables(max_positions, head_dim, rope_local)

    b = Builder()
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    attention_mask_input = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    type_ids = helper.make_tensor_value_info("type_ids", TensorProto.INT64, ["batch"])
    markers = helper.make_tensor_value_info("markers", TensorProto.INT64, ["n"])

    masks = attention_mask(b, "attention_mask")
    cos4_global, sin4_global = rope_terms(
        b, masks["positions"], cos_global, sin_global, head_dim, "global"
    )
    cos4_local, sin4_local = rope_terms(
        b, masks["positions"], cos_local, sin_local, head_dim, "local"
    )
    local_mask_name = local_mask(b, masks["distance"], masks["padding4"], local_window)

    embedding = b.gather(
        b.init(weights.get("encoder.embeddings.tok_embeddings.weight"), "tok_embeddings"),
        "input_ids",
        hint="embedded",
    )
    x = b.layer_norm(
        embedding,
        weights.get("encoder.embeddings.norm.weight"),
        "embeddings_norm",
        eps=eps,
    )

    for index in range(layers):
        hint = f"encoder_{index}"
        layer = f"encoder.layers.{index}"
        attention_norm = (
            b.layer_norm(
                x,
                weights.get(f"{layer}.attn_norm.weight"),
                f"{hint}.attn_norm",
                eps=eps,
            )
            if weights.has(f"{layer}.attn_norm.weight")
            else x
        )
        uses_local = index % global_every != 0
        mask = local_mask_name if uses_local else masks["padding4"]
        cos4, sin4 = (
            (cos4_local, sin4_local) if uses_local else (cos4_global, sin4_global)
        )
        attention = encoder_attention(
            b,
            attention_norm,
            weights.get(f"{layer}.attn.Wqkv.weight"),
            weights.get(f"{layer}.attn.Wo.weight"),
            heads,
            head_dim,
            mask,
            cos4,
            sin4,
            hint,
        )
        x = b.binary("Add", x, attention)

        mlp_input = b.layer_norm(
            x,
            weights.get(f"{layer}.mlp_norm.weight"),
            f"{hint}.mlp_norm",
            eps=eps,
        )
        mlp = b.linear(
            mlp_input,
            weights.get(f"{layer}.mlp.Wi.weight"),
            f"{hint}.mlp_in",
        )
        first, second = b.op(
            "Split",
            [mlp, b.i64s([intermediate, intermediate], "mlp_split")],
            outputs=2,
            axis=-1,
            hint=f"{hint}.mlp_parts",
        )
        gated = b.binary("Mul", b.gelu_erf(first), second)
        mlp = b.linear(gated, weights.get(f"{layer}.mlp.Wo.weight"), f"{hint}.mlp_out")
        x = b.binary("Add", x, mlp)

    x = b.layer_norm(
        x,
        weights.get("encoder.final_norm.weight"),
        "final_norm",
        eps=eps,
    )
    type_embedding = b.gather(
        b.init(weights.get("type_emb.weight"), "type_emb"),
        "type_ids",
        hint="type_embedding",
    )
    type_embedding = b.op(
        "Unsqueeze",
        [type_embedding, b.i64s([1], "type_axes")],
        hint="type_embedding_unsqueezed",
    )
    x = b.binary("Add", x, type_embedding)

    for index in range(head_layers):
        x = head_attention(
            b,
            x,
            weights,
            hidden,
            head_heads,
            head_dim_h,
            masks["head_mask4"],
            index,
        )

    flat = b.reshape(x, [-1, hidden], "hidden_flat")
    selected = b.gather(flat, "markers", hint="selected_markers")
    selected = b.layer_norm(
        selected,
        weights.get("scorer.0.weight"),
        "scorer_norm",
        weights.get("scorer.0.bias"),
        HEAD_NORM_EPS,
    )
    selected = b.linear(
        selected,
        weights.get("scorer.1.weight"),
        "scorer_in",
        weights.get("scorer.1.bias"),
    )
    selected = b.gelu_erf(selected)
    logits = b.linear(
        selected,
        weights.get("scorer.3.weight"),
        "scorer_out",
        weights.get("scorer.3.bias"),
    )
    logits = b.op(
        "Squeeze",
        [logits, b.i64s([1], "logits_axes")],
        hint="logits",
    )
    logits = b.alias(logits, "logits")

    return b.model(
        [input_ids, attention_mask_input, type_ids, markers],
        logits,
    )


def copy_model_assets(checkpoint: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    agent_source = checkpoint / "rl_agent_config.json"
    agent_target = output / "rl_agent_config.json"
    if agent_source.resolve() != agent_target.resolve():
        shutil.copy2(agent_source, agent_target)
    tokenizer = checkpoint / "tokenizer"
    tokenizer_target = output / "tokenizer"
    if tokenizer.is_dir():
        if tokenizer.resolve() != tokenizer_target.resolve():
            shutil.copytree(tokenizer, tokenizer_target, dirs_exist_ok=True)
    elif (checkpoint / "tokenizer.json").is_file():
        shutil.copy2(checkpoint / "tokenizer.json", output / "tokenizer.json")
    else:
        raise FileNotFoundError(f"no tokenizer found under {checkpoint}")
    sidecar = output / "onnx.json"
    if not sidecar.exists():
        sidecar.write_text(
            json.dumps(
                {
                    "model": "model.onnx",
                    "max_batch": 8,
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "type_ids": "type_ids",
                    "markers": "markers",
                    "output": "logits",
                },
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Laya safetensors checkpoint")
    parser.add_argument("output", type=Path, help="output directory for the ONNX export")
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="write only model.onnx (do not copy tokenizer/config/sidecar)",
    )
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "model.onnx"
    onnx.save(model, target)
    if not args.skip_assets:
        copy_model_assets(args.checkpoint, args.output)
    print(f"wrote {target} ({target.stat().st_size / 1024 / 1024:.1f} MiB)")
    if not args.skip_assets:
        print(f"ready to serve: {args.output}")


if __name__ == "__main__":
    main()
