from transformers import GPT2Model
import torch

from onnx_utils import decompose_layer_norm, get_paths, save_input_json, export_onnx


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.use_cache = False
        self.model.config._attn_implementation = "eager"

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
            )
        return out.last_hidden_state


def export_gpt2_onnx(model_name, model_config, seq_len):
    model = GPT2Wrapper(model_config)
    model.eval()

    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    output_dir, onnx_path = get_paths(model_name)
    save_input_json(output_dir, model_name, dummy_input)

    m = export_onnx(model, dummy_input, onnx_path)
    m = decompose_layer_norm(m)

    import onnx
    onnx.save(m, onnx_path)
    assert "LayerNormalization" not in {n.op_type for n in m.graph.node}


if __name__ == "__main__":
    SEQ_LEN = 8
    model_config = GPT2Model.from_pretrained("gpt2")
    export_gpt2_onnx("gpt2", model_config, SEQ_LEN)
