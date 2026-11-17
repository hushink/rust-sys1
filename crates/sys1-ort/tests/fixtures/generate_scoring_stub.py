"""Generate the synthetic ONNX graph used by the ORT backend tests.

The graph implements the System One token-scoring contract without any real
model weights:

    input_ids      int64 [batch, seq]
    attention_mask int64 [batch, seq]
    type_ids       int64 [batch]
    markers        int64 [n]
    -> logits      float32 [n]
    -> pooled      float32 [batch]

`logits` is the input token id at each marker position, cast to float32;
`pooled` is token 0 of each row. That is enough to verify batching, padding,
marker indexing, pooled-output extraction, and act-head wiring without
shipping a large checkpoint.

Run with:

    uv run --with onnx --with numpy --python 3.13 python generate_scoring_stub.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    flat_shape = numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat_shape")
    pooled_index = numpy_helper.from_array(np.array([0], dtype=np.int64), "pooled_index")
    pooled_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "pooled_axes")
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    type_ids = helper.make_tensor_value_info(
        "type_ids", TensorProto.INT64, ["batch"]
    )
    markers = helper.make_tensor_value_info("markers", TensorProto.INT64, ["n"])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n"])
    pooled = helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch"])

    nodes = [
        helper.make_node("Reshape", ["input_ids", "flat_shape"], ["flat_ids"]),
        helper.make_node("Gather", ["flat_ids", "markers"], ["selected"], axis=0),
        helper.make_node("Cast", ["selected"], ["logits"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["input_ids", "pooled_index"], ["first_token"], axis=1),
        helper.make_node("Squeeze", ["first_token", "pooled_axes"], ["first_token_squeezed"]),
        helper.make_node(
            "Cast", ["first_token_squeezed"], ["pooled"], to=TensorProto.FLOAT
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "sys1_scoring_stub",
        [input_ids, attention_mask, type_ids, markers],
        [logits, pooled],
        initializer=[flat_shape, pooled_index, pooled_axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="sys1-ort-tests",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    target = Path(__file__).with_name("scoring_stub.onnx")
    onnx.save(model, target)
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
