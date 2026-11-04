# cuAlign

cuAlign is an independent implementation of the sparse optimization stage described by Xiang et al. It maps every node in a directed, unweighted source graph to a distinct node in a target graph and scores a mapping by the number of conserved directed arcs. It accepts supplied candidate pairs in place of the paper's embedding and nearest-neighbor stages. [cuAlign paper](https://doi.org/10.1145/3624062.3625129)

CUDA runs belief propagation and weighted matching. The CPU builds sparse overlap factors, completes partial matchings when needed, and verifies the edge score. The CPU build also implements the complete search for use as a reference. Both backends count reciprocal arcs separately and score conserved self-loops.

## Build and test

Use Linux or WSL with GNU Make and a C++ compiler. Python 3 runs the synthetic integration tests.

```sh
make -j2 test
```

For CUDA, install a toolkit compatible with your driver and set the toolkit path and GPU architecture. The example targets an RTX 4060 Ti.

```sh
make -j2 test-cuda CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89
```

## Run

```sh
./cualign.cuda --backend cuda \
  --g1 source.gw --g2 target.gw --candidates candidates.txt \
  --start start.align --iterations 10 --output result.align
```

Inputs are plain LEDA GW files with a single-token label per node. Each listed arc is read in the direction of its endpoints. Each candidate line contains a source label followed by one or more allowed target labels, separated by whitespace. The optional starting map has one source and one target label per line. The program adds those pairs to the candidate set and retains the starting map if its score beats all proposed maps. Without a starting map, the candidates must admit a complete injection. Choose a new output path on a filesystem that supports hard links.

For the CPU build, use `./cualign --backend cpu` with the same input and output options. Use `--backend verify` on the CUDA build to compare CPU and GPU beliefs and mappings. The default limit is 10 million overlap factors. Increase `--max-factors` only after sizing memory for the supplied candidate sets. The CUDA factor arrays use about 58 bytes per factor, plus the graph and matching allocations.

## Paper

Lizhi Xiang, Arif Khan, S. M. Ferdous, SR Aravind, and Mahantesh Halappanavar. 2023. *cuAlign: Scalable Network Alignment on GPU Accelerators*. SC-W 2023, pp. 747–755. [DOI](https://doi.org/10.1145/3624062.3625129) · [Author-hosted PDF](https://smferdous1.github.io/files/pdf/papers/Xiang%20et%20al_2023_cuAlign.pdf) · [BibTeX](CITATION.bib)
