## Official Code of "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration" - Accepted at ICML 2025

### Abstract

<div align="center">
  <a href="https://ddaedalus.github.io/=en" target="_blank">Andreas&nbsp;Kontogiannis</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://www.linkedin.com/in/konstantinos-papathanasiou-4bbb1b176/?originalSubdomain=gr" target="_blank">Konstantinos&nbsp;Papathanasiou</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://people.duke.edu/~ys267/">Yi&nbsp;Shen</a><sup>4</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.nl/citations?hl=en&user=R3y5dxMAAAAJ" target="_blank">Giorgos&nbsp;Stamou</a><sup>1</sup>  <b>&middot;</b> &ensp;
  <a href="https://www.michaelmzavlanos.org/" target="_blank">Michael Μ.&nbsp;Zavlanos</a><sup>4</sup>  <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=PBX9aQUAAAAJ&hl=en" target="_blank">George&nbsp;Vouros</a><sup>5</sup>  
  <br>
  <sup>1</sup> National Technical University of Athens <sup>2</sup> Archimedes AI &emsp; &emsp; <sup>3</sup> ETH Zurich &emsp; <br>
  <sup>4</sup> Duke University &emsp; <sup>5</sup> University of Piraeus &emsp; <br> <br> <br> 




![smpe](smpe)

> Learning to cooperate in distributed partially observable environments with no communication abilities poses significant challenges for multi-agent deep reinforcement learning (MARL). 
This paper addresses key concerns in this domain, focusing on inferring state representations from individual agent observations and leveraging these representations to enhance agents' exploration and collaborative task execution policies. To this end, we propose a novel state modelling framework for cooperative MARL, where agents infer meaningful belief representations of the non-observable state, with respect to optimizing their own policies, while filtering redundant and less informative joint state information. Building upon this framework, we propose the MARL SMPE algorithm. In SMPE, agents enhance their own policy's discriminative abilities under partial observability, explicitly by incorporating their beliefs into the policy network, and implicitly by adopting an adversarial type of exploration policies which encourages agents to discover novel, high-value states while improving the discriminative abilities of others. Experimentally, we show that SMPE outperforms state-of-the-art MARL algorithms in complex fully cooperative tasks from the MPE, LBF, and RWARE benchmarks.


#### LBF command line
```bash
python3 main.py --config=smpe_lbf --env-config=gymma with env_args.time_limit=50 env_args.key="Foraging-2s-9x9-3p-2f-coop-v2"
```

#### MPE command line
```bash
python3 main.py --config=smpe_mpe --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpread-v0"
```

#### RWARE command line
```bash
python3 main.py --config=smpe_lbf --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-4ag-hard-v1"
```
