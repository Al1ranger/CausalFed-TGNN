"""Stop this task's coordinator after its complete 50k/100k matrix is saved.

The disjoint large-data workers own 500k. No training result is removed.
"""
import bootstrap
import json
import time
import psutil
from data import ROOT

target=ROOT/'results/runs/100000_46_homogeneous_centralized_clean.json'
while True:
    try:
        ready=target.exists() and json.loads(target.read_text())['status']=='COMPLETE'
    except (OSError,json.JSONDecodeError):ready=False
    if ready:break
    time.sleep(3)
stopped=[]
for process in psutil.process_iter(['pid','cmdline']):
    args=process.info['cmdline'] or []
    if any(a.replace('\\','/').endswith('experimental_validation/run.py') for a in args) and '--scales' in args:
        values=args[args.index('--scales')+1:]
        if values[:3]==['50000','100000','500000']:
            process.terminate();stopped.append(process.pid)
(ROOT/'COORDINATOR_HANDOFF.json').write_text(json.dumps(dict(status='CENTRAL_50K_100K_SAVED',stopped_pids=stopped,
    reason='500k owned by disjoint workers; avoid duplicate training',trigger=str(target)),indent=2))
print('Central stage saved; disjoint 500k workers continue.',flush=True)
