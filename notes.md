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
