from transformers import BertModel
from pathlib import Path
import json
import onnx
import torch


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
    """Export BERT model to ONNX, decompose LayerNorm, save dummy input."""
    hf_model = BertModel.from_pretrained(hf_id)
    model = BertWrapper(hf_model)
    model.eval()

    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    root = Path.cwd()
    output_dir = root / "zk-torch" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model_name}_input.json", "w") as f:
        json.dump({"input_data": dummy_input.numpy().tolist()}, f)

    onnx_path = str(root / "onnx" / f"{model_name}.onnx")
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, opset_version=11)
    m = onnx.load(onnx_path)

    ops = {n.op_type for n in m.graph.node}
    print("Opset:", m.opset_import)
    print("Ops:", ops)
    assert "LayerNormalization" not in ops, (
        "opset 11 should never emit LayerNormalization — bump to >=17 if you see this"
    )


if __name__ == "__main__":
    SEQ_LEN = 32
    MODEL = "bert_tiny"  # flip to "bert_base" once tiny is green

    if MODEL == "bert_tiny":
        export_bert_onnx("bert_tiny", "prajjwal1/bert-tiny", SEQ_LEN)
    elif MODEL == "bert_base":
        export_bert_onnx("bert_base", "bert-base-uncased", SEQ_LEN)
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")
