# Just a script to start the onnxruntime.
# Done for education

import onnxruntime
import torch

torch.manual_seed(42)
example_inputs = torch.randn(1, 4)

onnx_inputs = [example_inputs.numpy(force=True)]
print(f"Input length: {len(onnx_inputs)}")
print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    "./tiny_mlp.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {
    input_arg.name: input_value
    for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
print(f"Outputs: {onnxruntime_outputs}")
