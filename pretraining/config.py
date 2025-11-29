

# SEEN_OPPO_POLICY = ["policy1","policy2","policy3","low_blood"]
SEEN_OPPO_POLICY = ["low_blood"]
UNSEEN_OPPO_POLICY = []

OPPO_INDEX = [0]
AGENT_INDEX = [1]

num_steps=100
batch_size=16
actor_lr=1e-4
critic_lr=1e-4
gamma=0.99
num_update_per_iter=10
clip_param=0.2
max_grad_norm=5.0

state_dim=13
action_dim=4
hidden_dim=64
device="cuda:0"