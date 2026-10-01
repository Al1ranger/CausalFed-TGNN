import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from data import prepare

def test_original_partition_and_unknown_exclusion():
    d=prepare(50000,42)
    ids=d['df'].transaction_id
    populations=[set(ids[d['split']==i]) for i in range(3)]
    assert not populations[0]&populations[1]
    assert not populations[0]&populations[2]
    assert not populations[1]&populations[2]
    assert not d['unknown'][d['split']<2].any()
    previous=d['previous']; rows=np.arange(len(ids))[:,None]
    assert ((previous==-1)|(previous<rows)).all()
    for a,b in [(0,1),(1,2)]:
        assert d['df'].loc[d['split']==a,'timestamp'].max()<d['df'].loc[d['split']==b,'timestamp'].min()
