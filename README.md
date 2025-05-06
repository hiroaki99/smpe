## Official Code of "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration" - Accepted at ICML 2025

### Abstract

!(smpe)

> Learning to cooperate in distributed partially observable environments with no communication abilities poses significant challenges for multi-agent deep reinforcement learning (MARL). 
This paper addresses key concerns in this domain, focusing on inferring state representations from individual agent observations and leveraging these representations to enhance agents' exploration and collaborative task execution policies.
To this end, we propose a novel state modelling framework for cooperative MARL, where agents infer meaningful belief representations of the non-observable state, with respect to optimizing their own policies, while filtering redundant and less informative joint state information. 
Building upon this framework, we propose the MARL SMPE$^2$ algorithm.  
In SMPE$^2$, agents enhance their own policy's discriminative abilities under partial observability, explicitly by incorporating their beliefs into the policy network, and implicitly by adopting an adversarial type of exploration policies which encourages agents to discover novel, high-value states while improving the discriminative abilities of others. Experimentally, we show that SMPE$^2$ outperforms state-of-the-art MARL algorithms in complex fully cooperative tasks from the MPE, LBF, and RWARE benchmarks.


#### LBF command line
```bash
python3 main.py --config=smpe_lbf --env-config=gymma with env_args.time_limit=50 env_args.key="Foraging-2s-9x9-3p-2f-coop-v2"
```

#### MPE command line
```bash
python3 main.py --config=smpe_mpe --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpread-v0""
```

#### RWARE command line
```bash
python3 main.py --config=smpe_lbf --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-4ag-hard-v1""
```
