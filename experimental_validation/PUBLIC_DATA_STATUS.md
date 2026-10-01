# Public data status

Empirical public-data evaluation: NOT YET EVALUATED.
The existing PaySim `.part` file is not accepted as a complete verified dataset.
The publisher repository identifies Kaggle as the sample distribution source:
https://github.com/EdgarLopezPhD/PaySim and
https://www.kaggle.com/datasets/ealaxi/paysim1 . The dataset page returned no
downloadable content through the available text browser. No alternate mirror
was substituted and no empirical result is claimed.

IEEE-CIS: BLOCKED / AUTHENTICATION AND COMPETITION TERMS REQUIRED. No credentials
or accepted competition access were provided or bypassed. Official schema source:
https://www.kaggle.com/c/ieee-fraud-detection/data . Transaction and identity tables
join by TransactionID; TransactionDT is a relative offset. Identity coverage is
incomplete and must not be silently converted to complete-case sampling.

`public_adapters.py` contains schema-checked adapters and equal-time-preserving
chronological partitions. Tests use tiny, explicitly constructed fixtures only;
they are software tests, not experiments on either public dataset. PaySim supports
origin/destination accounts; merchant/device/location/bank relations are not
invented. Balance outcomes and fraud flags are excluded from predictive input.
IEEE-CIS anonymized card/device attributes are not promoted to verified account
identities. Public-data graph modeling requires a separately documented protocol.
