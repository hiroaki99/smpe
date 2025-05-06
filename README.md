## Official Code of "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration" - Accepted at ICML 2025

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