import asyncio
import json
from pathlib import Path

import torch

import ezkl  # type: ignore

onnx_model = "onnx/tiny_mlp.onnx"
CALIBRATION_DATA_FILE = "ezkl/generated/cal_data.json"
SETTINGS_FILE = "ezkl/generated/settings.json"
CIRCUIT_FILE = "ezkl/generated/model.ezkl"
SRS_FILE = "ezkl/generated/kzg.srs"
VERIF_KEY = "ezkl/generated/vk.key"
PROV_KEY = "ezkl/generated/pk.key"
INPUT_DATA = "ezkl/generated/input.json"
WITNESS = "ezkl/generated/witness.json"
PROOF = "ezkl/generated/proof.json"

### Steps
# Generate calibration data
# Create an initial settings file containing settings of the model/circuit
# Refine the settings values by calibrating them by doing inferences with calibration data
# Compile the circuit
# If not alrady downloaded get the srs file corresponding to ezkl universal setup
# Create the verification and proving key from srs and circuit
# Obtain an input which inference will be verified
# Run inference with an input and get witness form intermediate values
# Generate a proof with witness, circuit, proving key, srs
# Verify the proof with srs and verification key

### Generate calibration data
# This is used to run the model multiple times so
# ezkl can gauge the range of values of intermediary states.
# Afterwards ezkl can compute a scaling factor for quantization.
# In ezkl quantization range is per model
# Create 20 samples of 4 features
torch.manual_seed(42)
if not Path(CALIBRATION_DATA_FILE).exists():
    cal_data = {"input_data": [torch.randn(4).tolist() for _ in range(20)]}
    with open(CALIBRATION_DATA_FILE, "w") as f:
        json.dump(cal_data, f)


if not Path(SETTINGS_FILE).exists():
    ### Generate settings
    # Initial circuit configuration: scale, number of rows ...
    # Produce settings.json file
    ezkl.gen_settings(onnx_model, SETTINGS_FILE)  # type: ignore
    ### Calibrate settings
    # Refine the initial settings from settings.json through the calibration data
    # "resources" parameter -> minimize circuit size. Can also be "accuracy"
    # to maximize precision of the quantized model
    ezkl.calibrate_settings(
        CALIBRATION_DATA_FILE, onnx_model, SETTINGS_FILE, "resources"
    )  # type: ignore


if not Path(CIRCUIT_FILE).exists():
    ### Compile the circuit
    ezkl.compile_circuit(onnx_model, CIRCUIT_FILE, SETTINGS_FILE)

if not Path(SRS_FILE).exists():
    ### Fetch the SRS (structured reference string), the trusted setup
    async def main():
        await ezkl.get_srs(settings_path=SETTINGS_FILE, srs_path=SRS_FILE)  # type: ignore

    asyncio.run(main())

if not Path(VERIF_KEY).exists() or not Path(PROV_KEY).exists():
    ### Create verification and proving key
    ezkl.setup(CIRCUIT_FILE, VERIF_KEY, PROV_KEY, SRS_FILE)  # type: ignore


if not Path(INPUT_DATA).exists():
    ### Input value for witness generation
    torch.manual_seed(0)
    witness_data = {"input_data": [torch.randn(4).tolist()]}
    with open(INPUT_DATA, "w") as f:
        json.dump(witness_data, f)

if not Path(WITNESS).exists():
    ### Generate the witness
    # Runs the circuit with input data
    # Records all the intermediate values to construct the proof
    ezkl.gen_witness(INPUT_DATA, CIRCUIT_FILE, WITNESS)  # type: ignore

if not Path(PROOF).exists():
    ### Generate the proof
    # This is long
    ezkl.prove(WITNESS, CIRCUIT_FILE, PROV_KEY, PROOF, SRS_FILE)

### Verify the proof
res = ezkl.verify(PROOF, SETTINGS_FILE, VERIF_KEY, SRS_FILE)  # type: ignore
print(f"Verification is successful?\n{res}")
