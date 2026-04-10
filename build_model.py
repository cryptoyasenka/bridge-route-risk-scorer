"""Rebuild bridge-route-risk-scorer.onnx deterministically."""
import onnx
from onnx import TensorProto, helper

OUTPUT_PATH = "bridge-route-risk-scorer.onnx"
NUM_FEATURES = 10

features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, NUM_FEATURES])
route_risk = helper.make_tensor_value_info("route_risk", TensorProto.FLOAT, [1, 1])

nodes = [
    helper.make_node("Relu", ["features"], ["h1"], name="relu1"),
    helper.make_node("ReduceSum", ["h1"], ["h2"], name="rs1", axes=[1], keepdims=1),
    helper.make_node("Sigmoid", ["h2"], ["route_risk"], name="sig1"),
]
graph = helper.make_graph(nodes, "bridge_route_risk_graph", [features], [route_risk])
model = helper.make_model(graph, producer_name="cryptoyasenka", producer_version="1.00",
                          opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 7
onnx.checker.check_model(model)
onnx.save(model, OUTPUT_PATH)
print(f"wrote {OUTPUT_PATH}")
