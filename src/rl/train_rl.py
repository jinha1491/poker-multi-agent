import torch
import torch.optim as optim
import numpy as np
import copy
import wandb
from src.rl.environment import SimplifiedPokerEnv
from src.rl.policy import PokerPolicy


def train_rl(
    num_episodes: int = 3000,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    snapshot_interval: int = 200,
    entropy_coef: float = 0.001,
):
    """
    Train the poker policy using REINFORCE with a self-play baseline
    and light entropy regularization.

    Background: an earlier run without entropy regularization achieved
    0.65-0.68 avg reward but converged to a policy that favored "check"
    broadly rather than learning sharper bucket-specific optimal actions
    (e.g., it didn't learn to fold clearly weak hands, despite fold
    being the clear optimal action there). This is a classic premature
    convergence to a "safe" local optimum in policy gradient methods.

    A stronger entropy_coef (0.01) tested previously destabilized
    training and lowered average reward (0.65). This run uses a much
    smaller entropy_coef (0.001) plus a lower learning rate (5e-4) and
    more episodes (3000) to encourage sufficient exploration early on
    without the instability of the earlier attempt.

    Self-play mechanism:
    Every `snapshot_interval` episodes, we freeze a copy of the current
    policy as the "opponent snapshot" and compare average reward against
    it on the same batch of states.
    """
    wandb.init(
        project="poker-rl",
        config={
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "snapshot_interval": snapshot_interval,
            "entropy_coef": entropy_coef,
            "hidden_dim": 128,
            "algorithm": "REINFORCE with self-play baseline + light entropy regularization",
        }
    )
    cfg = wandb.config

    env = SimplifiedPokerEnv()

    print("Loading policy network...")
    policy = PokerPolicy(hidden_dim=cfg.hidden_dim)

    optimizer = optim.Adam(policy.policy_head.parameters(), lr=cfg.learning_rate)

    opponent_snapshot = copy.deepcopy(policy.policy_head)
    opponent_snapshot.eval()

    reward_baseline = 0.0
    baseline_alpha = 0.05

    print(f"\nStarting RL training for {cfg.num_episodes} episodes...")

    episode_rewards = []

    for episode in range(1, cfg.num_episodes + 1):
        states = [env.sample_state() for _ in range(cfg.batch_size)]
        texts = [s["text"] for s in states]

        embeddings = policy.encode_state(texts)

        # --- current policy forward pass ---
        logits = policy.policy_head(embeddings)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample()
        log_probs = dist.log_prob(action_indices)

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
            opp_action_indices = torch.argmax(opp_probs, dim=-1)

            opp_rewards = []
            for i, state in enumerate(states):
                action = policy.ACTIONS[opp_action_indices[i].item()]
                reward = env.step(state, action)
                opp_rewards.append(reward)
            opp_rewards = torch.tensor(opp_rewards, dtype=torch.float)

        self_play_advantage = (rewards.mean() - opp_rewards.mean()).item()

        batch_mean_reward = rewards.mean().item()
        reward_baseline = (1 - baseline_alpha) * reward_baseline + baseline_alpha * batch_mean_reward

        advantages = rewards - reward_baseline

        # small entropy bonus — enough to prevent premature convergence
        # to a "safe" local optimum, without destabilizing training like
        # the earlier entropy_coef=0.01 experiment did
        entropy = dist.entropy().mean()

        # REINFORCE loss: -log_prob * advantage, with light entropy bonus
        loss = -(log_probs * advantages).mean() - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.policy_head.parameters(), max_norm=1.0)
        optimizer.step()

        episode_rewards.append(batch_mean_reward)

        if episode % cfg.snapshot_interval == 0:
            opponent_snapshot = copy.deepcopy(policy.policy_head)
            opponent_snapshot.eval()
            print(f"  Updated self-play snapshot at episode {episode}")

        if episode % 10 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            wandb.log({
                "episode": episode,
                "batch_mean_reward": batch_mean_reward,
                "recent_avg_reward": recent_avg,
                "self_play_advantage": self_play_advantage,
                "loss": loss.item(),
                "reward_baseline": reward_baseline,
                "entropy": entropy.item(),
            })

        if episode % 100 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:04d}/{cfg.num_episodes} | "
                  f"Avg Reward (last 50): {recent_avg:.4f} | "
                  f"Self-play advantage: {self_play_advantage:.4f} | "
                  f"Entropy: {entropy.item():.4f} | "
                  f"Loss: {loss.item():.4f}")

    torch.save(policy.policy_head.state_dict(), "src/rl/policy_head.pt")
    print("\nRL training complete. Policy head saved to src/rl/policy_head.pt")

    wandb.finish()
    return policy


if __name__ == "__main__":
    train_rl()
