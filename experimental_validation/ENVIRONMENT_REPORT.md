# Execution environment

```json
{
  "python": "3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]",
  "os": "Windows-11-10.0.26200-SP0",
  "cpu": "AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD",
  "logical_cpus": 12,
  "ram_bytes": 16312721408,
  "device": "cpu",
  "cuda_available": false,
  "cuda_version": null,
  "packages": {
    "torch": "2.14.0",
    "torch-geometric": "2.8.0.post1",
    "numpy": "2.5.3",
    "pandas": "3.0.1",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "pytest": "9.1.1",
    "psutil": "7.2.2"
  },
  "deterministic_algorithms": true,
  "torch_threads": 2,
  "reporting_package_versions": {
    "matplotlib": "3.11.2"
  }
}
```

CPU-only; two threads per process. Concurrent worker workloads affect wall time. Deterministic algorithms enabled; exact cross-platform reproducibility is not guaranteed.
