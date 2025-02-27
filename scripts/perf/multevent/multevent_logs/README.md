Results from using MultEvent profiler, with ini config set to only look at L3/MALL metrics for MI300X (on an Alola node).
- Command used: `sudo MultEvent gpu gpu-df --ini df_mi300_all.ini -t 7 -O ~/logs`
- `-t` for sampling time of 7s, suitable for most rocfft-bench runs (there should be other ways of sampling though)
- First number in filename is modified to be the batch size, iter when running multiple sets of bench commands, single for just one --ntrial 1 bench run                                                                                              
- Bench command run during sampling: `./clients/staging/rocfft-bench --length 8192 --precision single --transformType 0 --ntrial 10 --batchSize <batch>`
