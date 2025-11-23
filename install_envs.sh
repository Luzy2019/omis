#!/bin/bash


cd envs/lb_foraging_envs
pip install -e .
cd ../../

cd envs/multiagent_particle_envs
pip install -e .
cd ../../

cd envs/overcooked_ai_envs
pip install -e .
cd ../../