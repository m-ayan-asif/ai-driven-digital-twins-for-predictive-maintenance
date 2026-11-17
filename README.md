# Resource-Aware Digital Twin Deployment for Predictive Maintenance

This project implements and evaluates a C-MAPSS turbofan remaining-useful-life (RUL) predictor under three resource-placement scenarios: cloud-only, edge-only, and hybrid edge-cloud. The deployment scenarios reuse the same trained model and predictions, so differences in prediction accuracy are not attributed to placement.

Authors: Muhammad Ayan Asif (22i-1097) and Ruhab Ahmad (23i-0559), FAST-NUCES Islamabad.

The study is a reproducible software experiment, not a physical cloud/edge deployment. Network RTT, bandwidth, and edge compute scaling are scenario assumptions; only feature extraction and model execution time are locally microbenchmarked.

## Research question and contribution

How do cloud-only, edge-only, and hybrid placement compare with a static cloud-only baseline in RUL accuracy, response latency, and resource use when the prediction pipeline is held constant?

The contribution is a documented end-to-end FD001 baseline and an auditable simulator that reports prediction metrics alongside configured latency, compute, and transfer trade-offs. It does not introduce a new prognostics algorithm or claim production-level latency or billing results.

## Dataset

The experiment uses the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset, subset FD001. The official training, test, and final-cycle RUL files can be downloaded from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) or its [dataset archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip).

Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. The public archive contains a nested `CMAPSSData.zip`; extract the three FD001 files from the nested archive. Raw and processed data are excluded from version control. The experiment manifest records SHA-256 hashes for the three input files.

## Environment and reproduction

Validated with Python 3.14.6 on Windows, NumPy 2.5.5, scikit-learn 1.9.1, and Matplotlib 3.11.2. `requirements.txt` pins direct dependencies; `requirements-lock.txt` captures the complete tested environment.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m src.data_prep --input_dir data/raw --output_dir data/processed
python -m experiments.run_all
python -m src.plot_results --input results/summary.csv --output-dir figures
python -m unittest discover -s tests -v
```

`experiments.run_all` trains a Random Forest with seeds 13, 37, and 42, evaluates the official 100-engine FD001 test set, benchmarks local preprocessing and inference, and writes the following reproducibility artifacts:

- `results/raw/model_runs.csv` — per-seed Random Forest and mean-only baseline metrics.
- `results/raw/deployment_runs.csv` — per-seed deployment scenario metrics.
- `results/summary.csv` — mean and standard deviation across the three model seeds.
- `results/run_manifest.json` — dataset hashes, parameters, package versions, host information, and deployment assumptions.
- `figures/deployment_tradeoffs.png` — simulated p95 latency and normalized resource-cost index.

The main entry points are `src/data_prep.py`, `src/models/rul_model.py`, `src/deployment_sim/simulator.py`, and `experiments/run_all.py`.

## Implementation and experimental design

The preprocessing pipeline uses trailing windows of 30 engine cycles. Each example contains the three current operating settings plus the mean, standard deviation, and linear slope of each of the 21 sensor channels (66 features). Training RUL labels are capped at 125 cycles. Test metrics use the official final-cycle RUL label for each of the 100 test engines.

The primary predictor is `RandomForestRegressor` with 100 trees, maximum depth 16, minimum leaf size 2, and `max_features=0.8`. A mean-only `DummyRegressor` is included as a weak reference baseline. Deployment comparisons use the same trained model and test predictions.

| Scenario | RTT | Bandwidth | Placement assumption | Per-request input |
|---|---:|---:|---|---:|
| Cloud-only | 80 ms | 20 Mbps | Feature extraction and inference in cloud | 2,880 bytes |
| Edge-only | 1 ms | 20 Mbps | Local compute multiplier 2.0 | 0 bytes |
| Hybrid edge-cloud | 80 ms | 20 Mbps | Edge feature extraction multiplier 2.0; cloud inference | 264 bytes |

Scenario latency combines local microbenchmark timings with these fixed assumptions. Payload accounting assumes float32 encoding, counts uplink input payload only, and excludes the RUL response. Requests are evaluated independently; the simulator does not add concurrent request queueing. The dimensionless resource-cost index is

```text
0.5 * (scenario compute time / cloud-only compute time)
+ 0.5 * (scenario transmitted bytes / cloud-only transmitted bytes)
```

Cloud-only is normalized to 1.0. This index is not a dollar estimate or provider bill. The simulator does not model concurrent request queues, real network variability, hardware heterogeneity, or physical cloud/edge services.

## Actual results

Three Random Forest seeds produced mean test RMSE `15.0266 ± 0.5049` cycles and MAE `11.8558 ± 0.4167` cycles. The training-mean baseline produced RMSE `43.0670` and MAE `35.9000` for each seed.

| Scenario | RMSE (cycles) | MAE (cycles) | Simulated p95 latency | Compute per inference | Bytes per inference | Resource-cost index |
|---|---:|---:|---:|---:|---:|---:|
| Cloud-only | 15.0266 ± 0.5049 | 11.8558 ± 0.4167 | 111.3216 ± 0.4218 ms | 28.4776 ms | 2,880 | 1.0000 |
| Edge-only | 15.0266 ± 0.5049 | 11.8558 ± 0.4167 | 61.3393 ± 0.8435 ms | 56.9552 ms | 0 | 1.0000 |
| Hybrid edge-cloud | 15.0266 ± 0.5049 | 11.8558 ± 0.4167 | 110.3261 ± 0.4053 ms | 28.5206 ms | 264 | 0.5466 |

Accuracy is identical across placement rows by design because each row reuses the same model predictions. Under the configured scenario assumptions, edge-only has the lowest simulated p95 latency; hybrid sends 90.83% fewer input bytes than cloud-only. The hybrid result retains the assumed cloud RTT, and its resource-cost index is only a normalized proxy. Neither result establishes real infrastructure performance or monetary savings.

The IEEE conference manuscript is [`main.tex`](main.tex) and includes the result table and generated trade-off figure. All listed academic references in the manuscript are journal publications with DOIs.

## Repository

Source code and reproducibility artifacts: https://github.com/m-ayan-asif/ai-driven-digital-twins-for-predictive-maintenance

## Limitations

The experiment covers FD001 only, one Random Forest model, and a mean-only reference baseline. The cloud/edge parameters are not calibrated from deployed services, and no real cloud or edge hardware was measured. Future work should evaluate all C-MAPSS subsets, compare additional models, calibrate deployment parameters, and validate on physical infrastructure.
