import torch
import torch.optim as optim
import numpy as np
import copy
import wandb
from src.rl.environment import SimplifiedPokerEnv
from src.rl.policy import PokerPolicy


def train_rl(
    num_episodes: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    snapshot_interval: int = 200,
):
    """
    Train the poker policy using REINFORCE with a self-play baseline.

    Self-play mechanism:
    Every `snapshot_interval` episodes, we freeze a copy of the current
    policy as the "opponent snapshot." We compare the current policy's
    average reward against the snapshot's average reward on the same
    batch of states. If the current policy is beating its past self,
    it gets a bonus signal reinforcing that direction.

    This creates a genuine curriculum: the agent isn't just chasing a
    static reward function, it's continuously trying to outperform
    earlier versions of itself.
    """
    wandb.init(
        project="poker-rl",
        config={
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "snapshot_interval": snapshot_interval,
            "algorithm": "REINFORCE with self-play baseline",
        }
    )
    cfg = wandb.config

    env = SimplifiedPokerEnv()

    print("Loading policy network...")
    policy = PokerPolicy()

    # only optimize the trainable policy head — DistilBERT stays frozen
    optimizer = optim.Adam(policy.policy_head.parameters(), lr=cfg.learning_rate)

    # self-play opponent snapshot — starts as a copy of the initial policy
    opponent_snapshot = copy.deepcopy(policy.policy_head)
    opponent_snapshot.eval()

    # running baseline for variance reduction — exponential moving average
    reward_baseline = 0.0
    baseline_alpha = 0.05

    print(f"\nStarting RL training for {cfg.num_episodes} episodes...")

    episode_rewards = []

    for episode in range(1, cfg.num_episodes + 1):
        # sample a batch of states
        states = [env.sample_state() for _ in range(cfg.batch_size)]
        texts = [s["text"] for s in states]

        # get embeddings once (frozen DistilBERT), reuse for both
        # current policy and opponent snapshot
        embeddings = policy.encode_state(texts)

        # --- current policy forward pass ---
        logits = policy.policy_head(embeddings)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample()
        log_probs = dist.log_prob(action_indices)

        # compute rewards for the current policy's chosen actions
        rewards = []
        for i, state in enumerate(states):
            action = policy.ACTIONS[action_indices[i].item()]
            reward = env.step(state, action)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # --- opponent snapshot forward pass (self-play comparison) ---
        with torch.no_grad():
            opp_logits = opponent_snapshot(embeddings)
            opp_probs = torch.softmax(opp_logits, dim=-1)
            opp_action_indices = torch.argmax(opp_probs, dim=-1)  # opponent plays greedily

            opp_rewards = []
            for i, state in enumerate(states):
                action = policy.ACTIONS[opp_action_indices[i].item()]
                reward = env.step(state, action)
                opp_rewards.append(reward)
            opp_rewards = torch.tensor(opp_rewards, dtype=torch.float)

        # self-play advantage — how much better is current policy vs snapshot
        self_play_advantage = (rewards.mean() - opp_rewards.mean()).item()

        # update running baseline
        batch_mean_reward = rewards.mean().item()
        reward_baseline = (1 - baseline_alpha) * reward_baseline + baseline_alpha * batch_mean_reward

        # advantage = reward - baseline (variance reduction technique)
        advantages = rewards - reward_baseline

        # REINFORCE loss: -log_prob * advantage
        # we want to increase log_prob for actions with positive advantage
        # and decrease it for actions with negative advantage
        loss = -(log_probs * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.policy_head.parameters(), max_norm=1.0)
        optimizer.step()

        episode_rewards.append(batch_mean_reward)

        # update self-play snapshot periodically
        if episode % cfg.snapshot_interval == 0:
            opponent_snapshot = copy.deepcopy(policy.policy_head)
            opponent_snapshot.eval()
            print(f"  Updated self-play snapshot at episode {episode}")

        # logging
        if episode % 10 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            wandb.log({
                "episode": episode,
                "batch_mean_reward": batch_mean_reward,
                "recent_avg_reward": recent_avg,
                "self_play_advantage": self_play_advantage,
                "loss": loss.item(),
                "reward_baseline": reward_baseline,
            })

        if episode % 100 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:04d}/{cfg.num_episodes} | "
                  f"Avg Reward (last 50): {recent_avg:.4f} | "
                  f"Self-play advantage: {self_play_advantage:.4f} | "
                  f"Loss: {loss.item():.4f}")

    # save final trained policy head
    torch.save(policy.policy_head.state_dict(), "src/rl/policy_head.pt")
    print("\nRL training complete. Policy head saved to src/rl/policy_head.pt")

    wandb.finish()
    return policy


if __name__ == "__main__":
    train_rl()
