# Notes

## ezkl

### Steps for setup, proof generation and verification

-  Generate calibration data
  - This is used to run the model multiple times so ezkl can gauge the range of values of intermediary states
  - ezkl can then compute a scaling factor for quantization
  - In ezkl quantization range is per model, simpler but less precise
-  Create an initial settings file containing settings of the model/circuit
  - number of rows, lookup table range, number of rows, visibility of input...
-  Refine the settings values by calibrating them by doing inferences with calibration data
  - more calibration data more precise generally
  - ideally have calibration data representing the future input distribution
-  Compile the circuit
-  If not alrady downloaded get the srs file corresponding to ezkl universal setup 
  - ezkl uses halo2 and uses universal setup
  - srs depends on the number of rows of  the circuit
-  Create the verification and proving key from srs and circuit
-  Choose an input which inference will be verified
-  Run inference with an input and get witness form intermediate values
-  Generate a proof with witness, circuit, proving key, srs
-  Verify the proof with srs and verification key

### Make a model parameters as circuit input

- _param_visibility_ setting
- how?
  - either in the settings.json file
  - or in the `ezkl.gen_settings`
```python
run_args = ezkl.PyRunArgs()
run_args.param_visibility = "hashed/public"  # type: ignore
# run_args.param_visibility = "polycommit"  # kzg commitment, another possibility
ezkl.gen_settings(onnx_model, SETTINGS_FILE, run_args)  # type: ignore
```
- retrieve the hash
  - the hash is part of the generated witnesse
```python
with open(WITNESS) as f:
    commit_hash = json.load(f)["processed_params"]["poseidon_hash"]
    print(f"Commitment of the model params: {commit_hash}")
```
- possible values: public, private, fixed, hashed/public, hashed/private, polycommit
  - `Fixed`
    - weights are baked into the circuit 
    - fast but circuit is specific to these weights
  - `Private`
    - weigts are used as circuit inputs 
    - weights are secret to the verifier
    - larger circuit, slower proving, verifier can't check which specific model was used
  - `Public`
  - `Hashed/public`
    - hash of the weights is revealed to the verifier
    - more complex circuit but good to check against public commitment
  - `Hashed/private`
    - hash of the weights is not shared to the verifier
    - not really useful: like private but with more complexity
  - `polycommit`
    - weights are commited using KZG instead of hashing
    - pro: supports opening proofs, you can prove individual parts of the commited data
      - will be useful for proof aggregation if we run out of memory
    - cons: even more complex circuit
-

### Transformer block issues 

- ezkl does not support pytorch implemetation of transformer out of the box
  - reimplement a custom transformer block
- ezkl parser, tract, only support older version of ONNX
  - onnx op_set version <= 16
  - set dynamo=False for torch.onnx_export to force older versions


### Stats 

- number of attention heads do not change timings
- seq_len increases timings linearly
- to predict timings it's better to focus on logrows rather than param numbers
- proof size seem to be constant

## zk-torch, compiling gpt2

### Issue with exporting gpt2 into onnx

- New versions of module transformers are not compatible with zktorch
  - we need to use old version of onnx exporter 
    - because new version (Dynamo Exporter) handles `LayerNorm` natively
    - zk-torch does not handle `LayerNorm` and need a decomposition of this block
    - old onnx exporter does this decomposition
      - to activate it old version when calling `torch.onnx.export` set
      - `opset_version=13` 
      - `dynamo=False`
-  Two solutions 
  - Decompose `LayerNorm` manually into primitive operations 
    - this is done by the script `export_gpt2_manual_layernorm.py`
  - Pick an older version of `transformers`
    - this is done by the script `export_gpt2_old_transformers.py`
      - "onnx>=1.15.0",
      - "torch>=2.1.0",
      - "transformers==4.37.0",
      
### Determining scale factor and ptau

- Scale factor 
  - What it is
    - factor which is used to go from floats to finite fields
    - _quantized = round(real × 2^scale_factor_log)_
  - higher scale factor better precision -> larger quantized values -> more expensive ptau
- cq_range_log 
  - What it is?
    - lookup tables cover values up to _2^cq_range_log_
    - all quantized values must fit in this range
  - hence _log2(max_activation × 2^scale_factor_log) < cq_range_log_
- ptau or powers of τ
  - What it is?
    - trusted setup file containing the elliptic curve points _[τ^0]G, ..., [τ^n]G_
    - required by KZG commitment scheme
    - generated once via snarkjs and value of τ is supposed to be destroyed/forgotten
  - we need to know how many powers of tau is needed for the circuit
    - number of ptau needs must satisfy two constraints simultaneously
      - _nb_ptau_log > cq_range_log_ (lookup table size)
      - _nb_ptau_log > log2(circuit_rows)_ (circuit commitment size)
  - power of tau too big and this will take up too much memory
  - power of tau too small and we lose too much precision
- Calibration process
  - find max activation value with `calibration_script.py`
  - find acceptable scale factor_log with `test_scalefactor.py`
    - compare KL divergence on an output with different scaling factor
- Calibration rules 
  - cq_range_log > 2 * scale_factor_log
  - pow_len_log > cq_range_log
  - cq_range_lower_log ~~cq_range_log - 1
  - LayerNorm ε >= 1 / (scale_factor * round(ε * 2)

### GPT2 vs BERT 

  Why BERT works and GPT-2 doesn't

  Four reasons, roughly in order of how much they matter:

  1. Outlier features are a decoder-only / scale problem.
  The Dettmers et al. ("LLM.int8()") paper found that concentrated outlier features in the residual
  stream — a few channels with magnitudes 100-1000× larger than the rest — emerge only above ~6.7B
  parameters in decoder-only LMs. They're an artifact of how autoregressive decoders learn to route
  long-range information through specific feature dimensions. BERT is bidirectional and well below
  that threshold, so its activations stay in a tight magnitude range (~2-30 across all channels).
  GPT-2-124M is small but still decoder-only, and its outlier features, while milder than GPT-3, are
  already enough to push activation magnitudes to ~4000.

  2. Pre-LN vs Post-LN.
  - GPT-2: x + Attn(LN(x)) — LayerNorm before each sublayer. The residual stream accumulates over
  depth without ever being directly normalized.
  - BERT (original): LN(x + Attn(x)) — LayerNorm after each sublayer. The residual is renormalized at
   every step.

  Result: BERT's residual magnitudes are bounded by construction. GPT-2's grow with depth. After
  squaring in LayerNorm, this is a 10-50× difference.

  3. GELU variant.
  - GPT-2 uses tanh-approx GELU: 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³))) — exports as the
  multi-op subgraph with Pow(x, 3) we wrote replace_gelu.py for. The x³ is itself a multiplication
  producing SF³-scale intermediates.
  - BERT uses erf-based GELU: 0.5 · x · (1 + erf(x/√2)) — exports as a single Erf op, which zk-torch
  handles as a univariate lookup. Zero rewriter needed, no SF² or SF³ intermediate.

  This single difference removes one of the worst SF³-scale offenders for free.

  4. Attention mask format.
  - GPT-2: causal mask uses Where(mask, scores, -inf). The -inf constant exports as float32 max
  (3.4e38) — that's the cq=137 we saw, literally unrepresentable in fixed point.
  - BERT: padding mask is additive — scores + (-10000) · (1 - mask). The -10000 is a normal,
  representable constant, and when there's no padding (mask=ones), the term is zero.

  Our BertWrapper passes attention_mask=None, so HF generates ones internally and the mask is
  effectively absent. No -inf constant ever appears in the graph.

  ---
  Quick magnitude comparison

  ┌───────────────────────────┬─────────────────┬───────────────────────────────────┬───────────┐
  │            Op             │   GPT-2 worst   │          BERT estimated           │   bits    │
  │                           │                 │                                   │   saved   │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────┼───────────┤
  │ LayerNorm Pow(x,2)        │ (4000)² = 1.6e7 │ (30)² = 900                       │ ~14       │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────┼───────────┤
  │ GELU Pow(x,3) (tanh       │ (4000)³ =       │ (no equivalent — Erf is           │ ~all      │
  │ approx)                   │ 6.4e10          │ univariate)                       │           │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────┼───────────┤
  │ Attention mask constant   │ 3.4e38          │ 0 (or 10000)                      │ ~107      │
  └───────────────────────────┴─────────────────┴───────────────────────────────────┴───────────┘

  So you're not just slightly better off with BERT — you're categorically in a different regime where
   SF can stay reasonable, fixed-point precision works, and ptau-22-ish is realistic. GPT-2's
  problems aren't "transformer problems," they're "decoder-only-LM-with-outlier-features problems."
