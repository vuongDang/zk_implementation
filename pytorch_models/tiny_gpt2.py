from transformers import GPT2Model, GPT2Config
from gpt2 import export_gpt2_onnx

if __name__ == "__main__":
    SEQ_LEN = 1
    VOCAB_SIZE = 50257

    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=SEQ_LEN,
        n_embd=8,
        # Specific to pass the tiny_gpt2, it happens because SF is too small
        layer_norm_epsilon=0.1,
        n_head=1,
        n_layer=1,
    )
    model_config = GPT2Model(config)
    export_gpt2_onnx("tiny_gpt2", model_config, SEQ_LEN)
