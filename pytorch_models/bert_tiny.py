from transformers import BertModel
import torch

from onnx_utils import get_paths, save_input_json, export_onnx


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config._attn_implementation = "eager"

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=None,
                token_type_ids=None,
            )
        return out.last_hidden_state


def export_bert_onnx(model_name, hf_id, seq_len):
    model = BertWrapper(BertModel.from_pretrained(hf_id))
    model.eval()

    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    output_dir, onnx_path = get_paths(model_name)
    save_input_json(output_dir, model_name, dummy_input)

    m = export_onnx(model, dummy_input, onnx_path)
    assert "LayerNormalization" not in {n.op_type for n in m.graph.node}, (
        "opset 11 should never emit LayerNormalization — bump to >=17 if you see this"
    )


if __name__ == "__main__":
    SEQ_LEN = 32
    MODEL = "bert_tiny"  # flip to "bert_base" once tiny is green
    export_bert_onnx(MODEL, "prajjwal1/bert-tiny", SEQ_LEN)
