# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import envs
import gymnasium as gym
import numpy as np
import time
import wandb

from methods.sac import SAC
from pyvirtualdisplay import Display
from utils.experiment import make_env, parse_args, setup_run


def train(args, exp_name, wandb_run, artifact):
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed, 0, args.capture_video, exp_name)]
        * args.num_envs
    )
    agent = SAC(args, envs.single_observation_space, envs.single_action_space)

    start_time = time.time()
    obs, _ = envs.reset()
    log = {}
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(args.num_envs)]
            )
        else:
            actions = agent.get_action(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        keys_to_log = [x for x in infos[0].keys() if x.startswith("reward_")]
        valuable_infos = {key: [] for key in keys_to_log}

        if "final_info" in infos:
            for info in infos["final_info"]:
                for key in keys_to_log:
                    valuable_infos[key].append(info[key])
            for key in keys_to_log:
                log[f"ep_info/{key.replace('reward_', '')}"] = np.mean(
                    valuable_infos[key]
                )
                print(
                    f"global_step={global_step}, episodic_return={np.mean(valuable_infos['reward_total'])}"
                )
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            update_actor = global_step % args.policy_frequency == 0
            policy_loss, qf1_loss, qf2_loss, alpha_loss = agent.update(
                args.batch_size, update_actor
            )

            if global_step % args.target_network_frequency == 0:
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                log.update(
                    {
                        "losses/Value1_loss": qf1_loss.item(),
                        "losses/Value2_loss": qf2_loss.item(),
                        "losses/alpha": agent.alpha,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                )

                if update_actor:
                    log.update({"losses/policy_loss": policy_loss.item()})
                if args.autotune:
                    log.update({"losses/alpha_loss": alpha_loss.item()})

        wandb.log(log, global_step)
        if global_step % 9999 == 0:
            agent.save(f"models/{exp_name}/")

    artifact.add_file(f"models/{exp_name}/actor.pt")
    wandb_run.log_artifact(artifact)
    envs.close()


def main(args):
    exp_name = f"{args.gym_id}_{int(time.time())}"
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    wandb_run, artifact = setup_run(exp_name, args)
    train(args, exp_name, wandb_run, artifact)


if __name__ == "__main__":
    args = parse_args()
    main(args)
