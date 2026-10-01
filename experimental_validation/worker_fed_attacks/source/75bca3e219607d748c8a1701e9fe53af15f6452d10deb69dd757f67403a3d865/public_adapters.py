"""Schema adapters only. No public-data empirical evaluation is implied."""
import bootstrap
import numpy as np
import pandas as pd

def time_split(values):
    """Keep equal-time groups together; achieved fractions may differ from 50/20/30."""
    s=pd.Series(values)
    if s.isna().any():raise ValueError('Missing event time')
    a,b=np.quantile(s,[.5,.7],method='higher')
    if a==b:raise ValueError('Insufficient temporal support for three disjoint windows')
    result=np.where(s<a,0,np.where(s<b,1,2))
    if len(np.unique(result))!=3:raise ValueError('Empty chronological partition')
    return result

def paysim(df):
    required={'step','type','amount','nameOrig','nameDest','isFraud'}
    if missing:=required-set(df):raise ValueError(f'Missing PaySim fields: {sorted(missing)}')
    if df[list(required)].isna().any().any():raise ValueError('Missing required input; no silent row deletion')
    if not set(df.isFraud.unique())<={0,1}:raise ValueError('Nonbinary target')
    out=df[['step','type','amount','nameOrig','nameDest','isFraud']].copy()
    out['source_row']=np.arange(len(df));out=out.sort_values(['step','source_row'],kind='stable')
    out['split']=time_split(out.step)
    # Explicitly exclude post-event balances and operational fraud flags.
    return dict(events=out,features=['type','amount'],entities=['nameOrig','nameDest'],time='step',target='isFraud',
                unsupported=['bank','device','location','unknown_mechanism'],
                exclusions=['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud'])

def ieee_cis(transaction,identity=None):
    required={'TransactionID','TransactionDT','TransactionAmt','isFraud'}
    if missing:=required-set(transaction):raise ValueError(f'Missing IEEE-CIS fields: {sorted(missing)}')
    if not transaction.TransactionID.is_unique:raise ValueError('Duplicate transaction IDs')
    if transaction[list(required)].isna().any().any():raise ValueError('Missing required field')
    if not set(transaction.isFraud.unique())<={0,1}:raise ValueError('Nonbinary target')
    out=transaction.copy()
    if identity is not None:
        if not identity.TransactionID.is_unique:raise ValueError('Duplicate identity keys')
        out=out.merge(identity,on='TransactionID',how='left',validate='one_to_one')
    out=out.sort_values(['TransactionDT','TransactionID'],kind='stable');out['split']=time_split(out.TransactionDT)
    return dict(events=out,time='TransactionDT',time_semantics='relative offset, not a wall-clock timestamp',
                target='isFraud',features=['TransactionAmt'],entities=[],
                unsupported=['verified bank IDs','verified account identity','unknown mechanism labels'])
