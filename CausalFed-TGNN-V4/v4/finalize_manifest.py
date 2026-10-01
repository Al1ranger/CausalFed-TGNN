"""Collect completed scale-run records into one auditable manifest."""
from pathlib import Path
import json, platform, sys

root=Path(__file__).resolve().parents[1]/"runs"/"v4_scale"
records=[]
for path in sorted(root.glob("synthetic_*.json")):
    if path.name == "RESULTS_MANIFEST.json": continue
    records.append(json.loads(path.read_text(encoding="utf-8")))
manifest={
    "v4_status":"scale_protocol_complete_neural_pending",
    "python":sys.version,
    "platform":platform.platform(),
    "completed_records":len(records),
    "scales":sorted({r["scale"] for r in records}),
    "seeds":sorted({r["seed"] for r in records}),
    "results":records,
    "claims":{
        "neural_performance":"pending_torch_environment",
        "public_paysim":"download_in_progress_or_pending_graph_run",
        "public_ieee_cis":"pending_kaggle_authentication",
        "causal_discovery":"not_implemented",
        "privacy_guarantee":"not_implemented"
    }
}
(root/"RESULTS_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps({"completed_records":len(records),"scales":manifest["scales"],"seeds":manifest["seeds"]},indent=2))
