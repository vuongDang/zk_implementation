from transformers import GPT2Model, GPT2Config
from transformers.generation.utils import EpsilonLogitsWarper
from gpt2 import  export_gpt2_onnx

SEQ_LEN = 4
VOCAB_SIZE = 100

config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=SEQ_LEN,
    n_embd=8,
    # Specific to pass the tiny_gpt2
    layer_norm_epsilon=1e-2,
    n_head=1,
    n_layer=1,
)
model_config = GPT2Model(config)
export_gpt2_onnx("tiny_gpt2", model_config, VOCAB_SIZE, SEQ_LEN)
