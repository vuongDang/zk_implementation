import asyncio
import json
from pathlib import Path
import onnx

import torch

import ezkl  # type: ignore

CALIBRATION_DATA_FILE = None
SETTINGS_FILE = None
CIRCUIT_FILE = None
SRS_FILE = None
VERIF_KEY = None
PROV_KEY = None
INPUT_DATA = None
WITNESS = None
PROOF = None


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


def setup(model, d_model, batch_size, seq_len):
    global \
        ONNX_MODEL, \
        CALIBRATION_DATA_FILE, \
        SETTINGS_FILE, \
        CIRCUIT_FILE, \
        SRS_FILE, \
        VERIF_KEY, \
        PROV_KEY, \
        INPUT_DATA, \
        WITNESS, \
        PROOF

    generated_dir = f"ezkl/generated/{model}_{d_model}"
    ONNX_MODEL = f"onnx/{model}.onnx"
    CALIBRATION_DATA_FILE = f"{generated_dir}/cal_data.json"
    SETTINGS_FILE = f"{generated_dir}/settings.json"
    CIRCUIT_FILE = f"{generated_dir}/model.ezkl"
    VERIF_KEY = f"{generated_dir}/vk.key"
    PROV_KEY = f"{generated_dir}/pk.key"
    INPUT_DATA = f"{generated_dir}/input.json"
    WITNESS = f"{generated_dir}/witness.json"
    PROOF = f"{generated_dir}/proof.json"
    Path(generated_dir).mkdir(parents=True, exist_ok=True)

    ### Generate calibration data
    # This is used to run the model multiple times so
    # ezkl can gauge the range of values of intermediary states.
    # Afterwards ezkl can compute a scaling factor for quantization.
    # In ezkl quantization range is per model
    # Create 20 samples of 4 features
    cal_data = {"input_data": [torch.randn(batch_size * seq_len * d_model).tolist() for _ in range(20)]}
    with open(CALIBRATION_DATA_FILE, "w") as f:
        json.dump(cal_data, f)

    ### Generate settings
    # Initial circuit configuration: scale, number of rows ...
    # Produce settings.json file
    # Set weights visibility to hashed public, to use them as input but give hash to verifier
    run_args = ezkl.PyRunArgs()
    run_args.param_visibility = "hashed/public"  # type: ignore
    # run_args.param_visibility = "polycommit"  # kzg commitment
    ezkl.gen_settings(ONNX_MODEL, SETTINGS_FILE, run_args)  # type: ignore

    ### Calibrate settings
    # Refine the initial settings from settings.json through the calibration data
    # "resources" parameter -> minimize circuit size. Can also be "accuracy"
    # to maximize precision of the quantized model
    ezkl.calibrate_settings(
        CALIBRATION_DATA_FILE, ONNX_MODEL, SETTINGS_FILE, "resources"
    )  # type: ignore

    ### Fetch the SRS (structured reference string), the trusted setup
    with open(SETTINGS_FILE) as f:
        logrows = json.load(f)["run_args"]["logrows"]

    SRS_FILE = f"ezkl/generated/kzg_{logrows}.srs"
    if not Path(SRS_FILE).exists():

        async def main():
            await ezkl.get_srs(settings_path=SETTINGS_FILE, srs_path=SRS_FILE)  # type: ignore

        asyncio.run(main())


def compile_circuit(reuse=True):
    if not reuse or not Path(CIRCUIT_FILE).exists():
        ### Compile the circuit
        ezkl.compile_circuit(ONNX_MODEL, CIRCUIT_FILE, SETTINGS_FILE)

    if not reuse or not Path(VERIF_KEY).exists() or not Path(PROV_KEY).exists():
        ### Create verification and proving key
        ezkl.setup(CIRCUIT_FILE, VERIF_KEY, PROV_KEY, SRS_FILE)  # type: ignore


def gen_proof(d_model, batch_len, seq_len, reuse=True) -> str:
    if not reuse or not Path(INPUT_DATA).exists():
        ### Input value for witness generation
        torch.manual_seed(0)
        witness_data = {"input_data": [torch.randn((batch_len, seq_len, d_model)).flatten().tolist()]}
        with open(INPUT_DATA, "w") as f:
            json.dump(witness_data, f)

    if not reuse or not Path(WITNESS).exists():
        ### Generate the witness
        # Runs the circuit with input data
        # Records all the intermediate values to construct the proof
        ezkl.gen_witness(INPUT_DATA, CIRCUIT_FILE, WITNESS)  # type: ignore

    if not reuse or not Path(PROOF).exists():
        ### Generate the proof
        # This is long
        ezkl.prove(WITNESS, CIRCUIT_FILE, PROV_KEY, PROOF, SRS_FILE)
    return str(PROOF)



def verif_proof() -> tuple[bool, list[str]]:
    ### Verify the proof
    res = ezkl.verify(PROOF, SETTINGS_FILE, VERIF_KEY, SRS_FILE)  # type: ignore
    # Hash of the weights can be retrieved
    with open(WITNESS) as f:
        commit_hash = json.load(f)["processed_params"]["poseidon_hash"]
    return (res, commit_hash)
