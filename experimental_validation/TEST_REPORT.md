# Test report

..............                                                           [100%]
============================== warnings summary ===============================
experimental_validation\packages\torch\jit\_script.py:1491
  D:\Study\Papers\p7\experimental_validation\packages\torch\jit\_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
14 passed, 1 warning in 23.73s


Independent artifact verification: PASS; 230 completed runs.

Verified source/configuration/checkpoint/prediction/log hashes and reconstructed known/open-set metrics. Neural CSV scores are restored to float32; historical float64 CSVs use round-trip parsing. All 15 raw stream hashes and chronological/D/E constraints passed. All 30 historical logistic records passed separate verification. Every completed neural checkpoint was reloaded and compared exactly on 128 test examples. Tests do not establish external validity, convergence, causal invariance, or privacy.
