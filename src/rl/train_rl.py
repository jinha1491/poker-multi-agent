import argparse
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
from src.rl.environment import SimplifiedPokerEnv
from src.rl.policy import PokerPolicy, state_features


def train_rl(seed=0, use_embedding=True, num_episodes=3000, batch_size=32,
             learning_rate=5e-4, snapshot_interval=200, entropy_coef=0.001):
    """
    REINFORCE with a moving-average baseline, self-play snapshots, and a
    light entropy bonus. Seeded so each run is reproducible, and run
    across several seeds to check convergence is reliable, not lucky.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = "emb_feat" if use_embedding else "feat_only"
    wandb.init(project="poker-rl", name=f"{tag}-seed{seed}", config={
        "seed": seed, "use_embedding": use_embedding, "num_episodes": num_episodes,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "snapshot_interval": snapshot_interval, "entropy_coef": entropy_coef,
        "hidden_dim": 128, "encoder": "poker_distilbert_v2" if use_embedding else None,
    })

    env = SimplifiedPokerEnv()
    policy = PokerPolicy(use_embedding=use_embedding)
    optimizer = optim.Adam(policy.policy_head.parameters(), lr=learning_rate)

    opponent_snapshot = copy.deepcopy(policy.policy_head)
    opponent_snapshot.eval()

    reward_baseline = 0.0
    baseline_alpha = 0.05
    episode_rewards = []

    print(f"Training {tag}, seed {seed}, {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        states = [env.sample_state() for _ in range(batch_size)]
        feats = torch.stack([state_features(s["hand_strength"], s["position"], s["stack_depth"])
                             for s in states])
        x = policy.build_input([s["text"] for s in states], feats)

        logits = policy.policy_head(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        rewards = torch.tensor([env.step(s, policy.ACTIONS[a.item()])
                                for s, a in zip(states, actions)], dtype=torch.float)

        # self-play: compare against a frozen earlier copy playing greedily
        with torch.no_grad():
            opp_actions = opponent_snapshot(x).argmax(dim=-1)
            opp_rewards = torch.tensor([env.step(s, policy.ACTIONS[a.item()])
                                        for s, a in zip(states, opp_actions)], dtype=torch.float)
        self_play_advantage = (rewards.mean() - opp_rewards.mean()).item()

        batch_mean = rewards.mean().item()
        reward_baseline = (1 - baseline_alpha) * reward_baseline + baseline_alpha * batch_mean
        advantages = rewards - reward_baseline

        entropy = dist.entropy().mean()
        loss = -(log_probs * advantages).mean() - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.policy_head.parameters(), max_norm=1.0)
        optimizer.step()

        episode_rewards.append(batch_mean)

        if episode % snapshot_interval == 0:
            opponent_snapshot = copy.deepcopy(policy.policy_head)
            opponent_snapshot.eval()

        if episode % 10 == 0:
            wandb.log({"episode": episode, "batch_mean_reward": batch_mean,
                       "recent_avg_reward": np.mean(episode_rewards[-50:]),
                       "self_play_advantage": self_play_advantage,
                       "entropy": entropy.item(), "loss": loss.item()})

        if episode % 250 == 0:
            print(f"Episode {episode:04d} | Avg Reward (last 50): {np.mean(episode_rewards[-50:]):.4f} "
                  f"| Entropy: {entropy.item():.4f}")

    # average over the last 200 episodes is a steadier number than the last 50
    final = float(np.mean(episode_rewards[-200:]))
    path = f"src/rl/policy_head_{tag}_seed{seed}.pt"
    torch.save(policy.policy_head.state_dict(), path)
    print(f"\nFINAL {tag} seed {seed}: avg reward (last 200 episodes) = {final:.4f}")
    print(f"Saved to {path}")

    wandb.summary["final_avg_reward"] = final
    wandb.finish()
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", action="store_true")
    args = ap.parse_args()
    train_rl(seed=args.seed, use_embedding=not args.no_embedding)
