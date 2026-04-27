# Script to verify onnx files
import onnx

model = onnx.load("onnx/transformer_block_dmod_16_nheads_1_dffn_32.onnx")
onnx.checker.check_model(model)
print("ONNX valid")
print("Opset:", model.opset_import[0].version)
ops = set(n.op_type for n in model.graph.node)
print("Ops used:", ops)

print(onnx.helper.printable_graph(model.graph))
