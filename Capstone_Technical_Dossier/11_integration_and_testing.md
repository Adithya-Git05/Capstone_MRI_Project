# 11 — Integration and testing

## Interfaces and errors

- `mri_io.write_uploaded_study` writes isolated NIfTI uploads and `validate_study` checks four required modalities, 3D arrays, finite/nonempty data, matching shape/affine and usable spacing.
- `pipeline.run_pipeline` executes 3D segmentation, validates output orientation/mask geometry, extracts a tumor-centre or manual slice, then optionally invokes 2D models. It records per-stage status; missing risk inputs are explicitly blocked.
- `risk_adapter.run_v8_risk_mapping` requires user-supplied SynthSeg labels, matching MRI geometry and allowed integer labels, reorients to RAS, and runs legacy V8 in a temporary directory. Integrated route does not supply DTI/tract masks.
- `model_adapters` loads checkpoints using `torch.load(weights_only=True)` and validates expected checkpoint structure/class names; inference failures become explicit stage errors.
- `rag_adapter.query_analysis_context` validates URL scheme, case/context match and question length; remote non-HTTPS is rejected. It posts structured context and checks returned citations against retrieved evidence.
- UI catches expected input/runtime/import exceptions and displays explicit failure state/message.

## Tests present

Current workspace test sources cover classification, attention segmentation, 3D ontology, smoke/data behavior, integrated input/spatial contracts/report output, and RAG API/core behavior. `test_inventory.csv` counts test functions in source files (24 root module tests, 15 integration tests, 23 RAG tests); these are definitions, not executed test results. No tests were run for dossier creation and no archived pass/fail summary was found.

## Integration limitations

The app requires four registered NIfTIs and a compatible 3D checkpoint. Optional BRISC models run on MRI slices out of domain. Risk is optional and requires registered SynthSeg; it is not inferred from the BraTS segmentation. DTI and tract scores unavailable in the current adapter. RAG requires an external local/authorized API and Ollama configuration. UI uses temporary run files and session state; no persistent result database is documented. No recorded complete end-to-end execution or clinical user evaluation exists.
