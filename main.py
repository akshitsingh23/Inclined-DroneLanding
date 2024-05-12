from Environments.crazyflie import Crazyflie_3d_setpoint, Crazyflie_2d_inclined
import numpy as np
import torch
from Reward.rewardfuncs import sparse_reward2d, euclidean_reward3d
from math import pi
from Save_Gif.save_gif import save_frames_as_gif
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, A2C, TD3
import random
import os
import matplotlib.pyplot as plt

# Use the GPU for training
if not torch.cuda.is_available():
    raise NotImplementedError("No GPU")
else:
    device = torch.device('cuda')

if __name__ == '__main__':

    environment = 'CF_2d_inclined'
    algorithm = 'PPO'               # PPO is fast, robust, and on-policy for curriculum learning
    training_timesteps = 3000000    # Total amount of Timesteps to train for
    pwm = 16000                     # PWM from theoretical hover Pwm, in the minus or plus direction.
    t_s = 1/50                      # seconds
    angles_vector=[-pi/5,-pi/6,-pi/7,-pi/8,-pi/9]
    angle=-pi/6
    reward_log=[]
    obs_log=[]
    
    for angle in angles_vector:
        if environment == 'CF_2d_inclined':
            env=Crazyflie_2d_inclined(landing_angle=angle,t_s=t_s, rewardfunc=sparse_reward2d, max_pwm_from_hover=pwm)
            
        #env = Crazyflie_2d_inclined(landing_angle=pi/6,t_s, rewardfunc=sparse_reward2d, max_pwm_from_hover=pwm)

        elif environment == 'CF_3d_setpoint':
            env = Crazyflie_3d_setpoint(t_s, rewardfunc=euclidean_reward3d, max_pwm_from_hover=pwm)

        # Check if the environment is working right
        # check_env(env)

        # Set seeds to be able to reproduce results
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        env.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Select the algorithm from Stable Baselines 3
        if algorithm == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed) # Try 0.995 for inclined landing
        elif algorithm == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, seed=seed)
        elif algorithm == 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)

        model.learn(training_timesteps)

        # Name the pytorch model
        run_name = environment+"_"+algorithm+"_"+str(training_timesteps)+"Timesteps"

        # Save the pytorch model
        torch.save(model.policy.state_dict(), run_name + 'state_dict.pt')

        # Render and save the gif afterwards
        obs = env.reset()
        frames = []

        # Render
        for i in range(1800):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_log.append(reward)
            obs_log.append(obs)
            frames.append(env.render(mode='rgb_array'))
            if done:
                obs = env.reset()
        # Save Gif
        save_frames_as_gif(frames, filename=run_name+'.gif')
        env.close()
        #timestamps vs reward log
        # plt.plot(reward_log)
        # plt.show()
        with open('rewards.txt', 'w') as f:
            for reward in reward_log:
                f.write(f"{reward}\n")  # Writing each reward on a new line

    # Writing observations to a text file
        with open('observations.txt', 'w') as f:
            for obs in obs_log:
                obs_str = ', '.join(map(str, obs))  # Convert each observation sublist to a string
                f.write(f"{obs_str}\n")
                
                #later extract the file and plot from the data
        
        
        # Reading rewards from text file
        with open('rewards.txt', 'r') as f:
            reward_log = [float(line.strip()) for line in f]

        # Reading observations from text file
        with open('observations.txt', 'r') as f:
            obs_log = [list(map(float, line.strip().split(', '))) for line in f]

        print("angle reward:",reward_log)
        print("angle obs:",obs_log)

    
# Plotting rewards
# plt.figure(figsize=(10, 5))
# plt.plot(reward_log, marker='o', linestyle='-')
# plt.title('Reward Over Time')
# plt.xlabel('Time step')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.show()


# # Extracting individual components
# x_vals = [obs[0] for obs in obs_log]
# y_vals = [obs[1] for obs in obs_log]

# # Plotting observations as time series
# plt.figure(figsize=(10, 5))
# plt.plot(x_vals, label='X Values', marker='o', linestyle='-')
# plt.plot(y_vals, label='Y Values', marker='o', linestyle='--')
# plt.title('Observations Over Time')
# plt.xlabel('Time step')
# plt.ylabel('Observation Value')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optionally, plotting x vs y if it makes sense (like plotting a trajectory)
# plt.figure(figsize=(6, 6))
# plt.plot(x_vals, y_vals, marker='o', linestyle='-')
# plt.title('Y vs X')
# plt.xlabel('X Value')
# plt.ylabel('Y Value')
# plt.grid(True)
# plt.show()






