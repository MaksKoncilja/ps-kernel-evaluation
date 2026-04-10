# ps-kernel-evaluation

Trying PS in combination with HMC / LMC (ULA, MALA) / NUTS / MCLMC / etc, and comparing that to SMC and perhaps Parallel Tempering.

See `PS.ipynb` for a better description of the project.

## done:

- `targets.py`  
  prior, likelihood, posterior

- `samplers.py`  
  wrappers for PS + RW / HMC / LMC (ULA, MALA) / NUTS / MCLMC / etc

- `reference.py`  
  reference SMC + RWM kernel (loop over dim, # particles, # of MCMC steps, seeds)

- `experiment.py`  
  loops over: dim, # particles, kernel, # of MCMC steps, seeds

## in the workings:

- `metrics.py`  
  metrics calculation from raw and summary data

- `plots.py`  
  plots results for kernel/model comparison
