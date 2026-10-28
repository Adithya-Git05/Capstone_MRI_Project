# 5. Surgical-risk mapping

The risk map and corridor implementation is in this stage's `pipeline/`,
`scripts/`, and `configs/`. Sample and local patient inputs are under this
stage's `data/`, `Risk_input/`, and `synthseg/`.

From the repository root, the V8 module uses this stage's
`configs/config_v7.json` by default:

```powershell
python -m pipeline.run_v8_pipeline
```

Before running, edit `5_surgical_risk_mapping/configs/config_v7.json`'s input paths for the selected case and
confirm that the MRI, tumour mask, SynthSeg labels, and optional tract masks
share the expected geometry. The example config references local sample files
and may need adjustment for another checkout. Keep all patient scans and
generated NIfTI maps local. See the project dossier and
[`docs/surgical_risk_implementation_notes.docx`](docs/surgical_risk_implementation_notes.docx)
for required inputs and known gaps.

These scores and corridor candidates are research outputs, not probabilities
or a surgical plan. No risk-stage command is presented here as verified
end-to-end until its inputs and output are checked for the intended case.
