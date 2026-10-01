import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import pandas as pd
import pytest
from public_adapters import paysim,ieee_cis,time_split

def test_paysim_does_not_invent_entities_or_use_outcomes():
    f=pd.DataFrame(dict(step=list(range(10)),type=['PAYMENT']*10,amount=[1]*10,nameOrig=['a']*10,nameDest=['b']*10,isFraud=[0,1]*5,newbalanceOrig=[0]*10,isFlaggedFraud=[0,1]*5))
    a=paysim(f)
    assert a['entities']==['nameOrig','nameDest'] and 'newbalanceOrig' not in a['events']
    assert len(a['events'])==len(f) and list(f.columns)!=list(a['events'].columns)
    with pytest.raises(ValueError):paysim(f.drop(columns='nameOrig'))

def test_equal_times_do_not_cross_splits():
    values=pd.Series([1,1,2,2,3,3,4,4,5,5]);s=time_split(values)
    for t in values.unique():assert len(set(s[values==t]))==1

def test_ieee_join_retains_missing_identity():
    tx=pd.DataFrame(dict(TransactionID=range(10),TransactionDT=range(10),TransactionAmt=[1]*10,isFraud=[0,1]*5))
    identity=pd.DataFrame(dict(TransactionID=[0],DeviceInfo=['d']))
    a=ieee_cis(tx,identity)
    assert len(a['events'])==10 and a['events'].DeviceInfo.isna().sum()==9 and a['entities']==[]
