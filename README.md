# Opponent Modeling with In-context Search (OMIS)

This is the code repository for the NeurIPS 2024 paper **Opponent Modeling with In-context Search**.

## Environment dependencies installation
Please follow the steps below to install environment dependencies. Installation on Ubuntu 18.04 LTS is recommended.
```bash
# using Anaconda to create a virtual environment
conda create --name omis python=3.8
conda activate omis

# install dependencies (might be tricky to install but necessary)
pip install -r requirements.txt

# install environments
./install_envs.sh
```

## Opponent policies download
Please download the *opponent policies* from the [this link](https://osf.io/6px85/?view_only=81c878f8fdfe4dbc90b8b21dae4e4d22). Then, place the downloaded opponent policies in the following path:
```bash
.
├── envs
│   ├── multiagent_particle_envs
│   │   ├── models
│   │   │   └── pbt_st
│   │   └── ...
├── ├── lb_foraging_envs
│   │   ├── models
│   │   │   └── pbt_lbf
│   │   └── ...
│   └── overcooked_ai_envs
│       ├── models
│       │   └── pbt_simple
│       └── ...
└── ...
```

## Pretraining

Train the Best Responses (BRs) against all the opponent policies in the training set of opponent policies. 
```bash
cd pretraining
python train_br.py
```

Generate the pretraining data by playing using BRs against all the opponent policies in the training set of opponent policies.
```bash
# generate the BR data
python gen_dataset.py
# generate the in-context data
python gen_prompt.py
``` 

Pretrain the Transformer model for decision-making based on in-context learning, where the model consists of three in-context components: an actor, an opponent imitator, and a critic.
```bash
python pretrain.py
```

## Testing
Search with the pretrained three in-context components to refine the original actor's policy.
```bash
cd ../testing
python test.py
```

Additionally, pretrained model weight files are available at `pretraining/models`.

## Remark
The opponent policies are trained using the codes of [maximum_entropy_population_based_training](https://github.com/ruizhaogit/maximum_entropy_population_based_training). The code for the Predator Prey (PP) environment is based on [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs). The code for the Level-Based Foraging (LBF) environment is based on [lb-foraging](https://github.com/semitable/lb-foraging/tree/master). The code for the OverCooked (OC) environment is based on [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai).
