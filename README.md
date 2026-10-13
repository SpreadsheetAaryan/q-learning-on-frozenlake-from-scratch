# Q-learning on FrozenLake from scratch

A tabular Q-learning agent for the FrozenLake environment in Gymnasium, written with NumPy: the Q-table, epsilon-greedy exploration with decay, the temporal-difference update, a training loop, and greedy evaluation.

This is a guided implementation from [Deep-ML](https://www.deep-ml.com): the project is broken into small steps, and I wrote each function in `model.py`. `scaffold.py` runs the whole pipeline end to end.

## What's implemented

- **Q-table** initialisation and lookups (`max_q_value`, `greedy_action`).
- **Exploration:** epsilon-greedy action selection with exponential epsilon decay.
- **Learning:** TD target, TD error, and the Q-learning update.
- **Training and evaluation:** episode loop, greedy policy extraction, and success rate over many greedy episodes.

## Run it

```bash
pip install -r requirements.txt
python scaffold.py
```

Sample output (CPU, a few seconds):

```
FrozenLake: 16 states, 4 actions
Q-table shape: (16, 4)
Greedy policy (first 8 states): [2, 2, 1, 0, 1, 0, 1, 0]
Single greedy episode reached goal: True
Greedy success rate over 100 episodes: 1.00
```

## Notes

- The greedy policy reaches the goal in 100 of 100 evaluation episodes on the deterministic map.

<details>
<summary>All implementation steps</summary>

- [x] **1.** init_q_table
- [x] **2.** max_q_value
- [x] **3.** greedy_action
- [x] **4.** sample_random_action
- [x] **5.** should_explore
- [x] **6.** epsilon_greedy_action
- [x] **7.** decay_epsilon
- [x] **8.** td_target
- [x] **9.** td_error
- [x] **10.** q_learning_update
- [x] **11.** interaction_step
- [x] **12.** run_training_episode
- [x] **13.** train_q_learning
- [x] **14.** extract_greedy_policy
- [x] **15.** run_greedy_episode
- [x] **16.** evaluate_success_rate

---

Built on Deep-ML.

</details>
