"""Generate the synthetic ONNX graph used by the ORT backend tests.

The graph implements the System One token-scoring contract without any real
model weights:

    input_ids      int64 [batch, seq]
    attention_mask int64 [batch, seq]
    type_ids       int64 [batch]
    markers        int64 [n]
    -> logits      float32 [n]

`logits` is the input token id at each marker position, cast to float32. That
is enough to verify batching, padding, marker indexing, and output extraction
without shipping a large checkpoint.

Run with:

    uv run --with onnx --with numpy --python 3.13 python generate_scoring_stub.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    flat_shape = numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat_shape")
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

    nodes = [
        helper.make_node("Reshape", ["input_ids", "flat_shape"], ["flat_ids"]),
        helper.make_node("Gather", ["flat_ids", "markers"], ["selected"], axis=0),
        helper.make_node("Cast", ["selected"], ["logits"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "sys1_scoring_stub",
        [input_ids, attention_mask, type_ids, markers],
        [logits],
        initializer=[flat_shape],
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
