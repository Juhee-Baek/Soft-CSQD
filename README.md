# Code for Soft Cluster-Adaptive Sample-Based Quantum Diagonalization: A Hybrid Quantum–Classical Algorithm Across Weak and Strong Correlation Regimes

This repository contains the source code used for the Soft Cluster-Adaptive Sample-Based Quantum Diagonalization (Soft-CSQD) workflow described in the associated manuscript. The code implements the clustering, recovery, subspace construnction, and projected diagonalization procedures used in the study.

## Folder structure

- `soft_csqd/csqd/`  
  Contains the core CSQD implementation, including configuration handling, sample processing, particle-number recovery, cluster-aware subsampling, selected-CI diagonalization, diagnostics, and result serialization.

- `soft_csqd/csqd/clustering/`  
  Contains the clustering modules used in this study. The supported methods are hard K-Modes, fuzzy K-Modes, hard Bernoulli mixture model (BMM), and soft BMM. Automatic cluster-number selection modules are not included in this manuscript-oriented version.

- `soft_csqd/examples/`  
  Contains helper scripts for chemistry-related input preparation used by the example workflows.

- `soft_csqd/run_n2_soft_csqd.py`  
  Provides the workflow runner for the N2 bond-dissociation calculations.

- `soft_csqd/run_2fe2s_soft_csqd.py`  
  Provides the workflow runner for the 2Fe-2S calculations.

- `soft_csqd/csqd_paper.py`  
  Provides a manuscript-oriented compatibility entry point for running the soft-CSQD workflow through the original paper-style API.

- `requirements.txt`  
  Lists the Python packages required to run the code.

## Soft-clustering workflow

This archive focuses on controlled comparisons between hard and soft clustering variants. The number of clusters is specified explicitly by the user through the `--n_clusters` argument or by setting `ClusteringConfig(n_clusters=...)` in Python.

The following clustering modes are supported:

- `kmodes`  
  Hard K-Modes clustering baseline. Each sampled string is assigned to one cluster.

- `fuzzy_kmodes`  
  Fuzzy K-Modes clustering. Each sampled string is assigned fractional memberships across clusters.

- `bmm`  
  Bernoulli mixture model with hard argmax assignments by default.

- `bmm --soft_assignment`  
  Bernoulli mixture model using posterior responsibilities as soft memberships.

The auto-K and active-K components used for automatic cluster-number selection were removed from this version. Therefore, this archive is intended for matched comparisons of hard and soft clustering effects under user-specified cluster numbers.

## Results

The result files are generated when the example runners are executed. Output files are written using naming that includes the clustering method, soft-assignment setting, number of clusters, target system, subspace dimension, and sampling settings. Auto-K or active-K tags are not used.

Example output file names include:

```text
csqd_ref_fuzzy_kmodes_soft_k5_n2_d17_maxdim2000_spb2000.pkl
csqd_ref_bmm_soft_k5_n2_d17_maxdim2000_spb2000.pkl
csqd_ref_fuzzy_kmodes_soft_k5_2fe2s_2000_nb10_ns2000_it10.pkl
csqd_ref_bmm_soft_k5_2fe2s_2000_nb10_ns2000_it10.pkl
```

The output pickle files contain the CSQD result object, including the selected-CI result, clustering result, configuration metadata, and diagnostic information. For soft methods, the cluster-level membership matrix is stored in `result.cluster_result.membership`. For hard methods, this matrix is one-hot. For fuzzy K-Modes and soft BMM, this matrix contains fractional cluster memberships. Determinant-level membership information is stored in `result.best_result.membership` when available.

The results can be used to compare the variational energies obtained from hard and soft clustering methods under matched values of `K`, `max_dim`, `n_samples_per_batch`, `n_batch`, and `max_iterations`.

## Reproducibility

The files in this archive correspond to the soft-CSQD workflow used for manuscript-oriented calculations. To reproduce the main workflow:

1. Install the required Python packages listed in `requirements.txt`.
2. Prepare the molecular integral files and quantum-sample input files required by the target system.
3. Run `run_n2_soft_csqd.py` for N2 calculations or `run_2fe2s_soft_csqd.py` for 2Fe-2S calculations.
4. Specify the clustering method using `--method` and the number of clusters using `--n_clusters`.
5. Use `--soft_assignment` with `--method bmm` to run soft BMM.
6. Compare the generated result files under matched values of `K`, `max_dim`, `n_samples_per_batch`, `n_batch`, and `max_iterations`.

Example commands are shown below.

Hard K-Modes for N2:

```bash
python -m soft_csqd.run_n2_soft_csqd \
  --method kmodes \
  --distance 1.7 \
  --n_clusters 5 \
  --max_dim 2000 \
  --n_samples_per_batch 2000 \
  --n_batch 10 \
  --max_iterations 10
```

Fuzzy K-Modes for N2:

```bash
python -m soft_csqd.run_n2_soft_csqd \
  --method fuzzy_kmodes \
  --distance 1.7 \
  --n_clusters 5 \
  --fuzzifier 2.0 \
  --max_dim 2000 \
  --n_samples_per_batch 2000 \
  --n_batch 10 \
  --max_iterations 10
```

Soft BMM for 2Fe-2S:

```bash
python -m soft_csqd.run_2fe2s_soft_csqd \
  --method bmm \
  --soft_assignment \
  --n_clusters 5 \
  --max_dim 2000 \
  --n_samples_per_batch 2000 \
  --n_batch 10 \
  --max_iterations 10
```

## Citation

Please cite the associated article and the repository or archive DOI for this code package.


## License

This project is licensed under the Apache License 2.0. See the
[LICENSE](Apache-2.0_qiskit-addon-sqd_LICENSE.txt) file for details.