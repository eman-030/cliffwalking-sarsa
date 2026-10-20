"""Environment creation, training, independent evaluation and visual output."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from agent import SarsaAgent

EPISODES = 3000
MAX_STEPS = 500

def make_env(render_mode=None):
    # Explicit time limit because the underlying task can continue after falls.
    return gym.make('CliffWalking-v1', is_slippery=False,
                    max_episode_steps=MAX_STEPS, render_mode=render_mode)

def evaluate(agent, epsilon=0.0, episodes=300, seed=20000, random_policy=False):
    env = make_env()
    rng = np.random.default_rng(seed)
    returns, steps, falls, successes = [], [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total, count, n_falls = 0, 0, 0
        while True:
            if random_policy or rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                action = int(np.argmax(agent.q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            count += 1
            n_falls += int(reward == -100)
            if terminated or truncated:
                returns.append(total)
                steps.append(count)
                falls.append(n_falls)
                successes.append(terminated)
                break
    env.close()
    return {'mean_return':float(np.mean(returns)), 'return_std':float(np.std(returns, ddof=1)) if episodes>1 else 0.0,
            'success_rate':float(np.mean(successes)), 'mean_steps':float(np.mean(steps)),
            'mean_cliff_falls':float(np.mean(falls)), 'episodes':episodes,
            'evaluation_epsilon':epsilon, 'random_policy':random_policy, 'seed':seed}

def train(epsilon=0.1, seed=30, episodes=EPISODES, verbose=True):
    env = make_env()
    agent = SarsaAgent(seed=seed)
    rewards, falls, checkpoints = [], [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        action = agent.act(state, epsilon)
        total, n_falls = 0, 0
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = None if terminated else agent.act(next_state, epsilon)
            # A cliff fall returns to start; it is NOT a terminal transition.
            # A time limit ends the rollout but still permits bootstrapping.
            agent.update(state, action, reward, next_state, next_action, terminated)
            total += reward
            n_falls += int(reward == -100)
            if terminated or truncated:
                break
            state, action = next_state, next_action
        rewards.append(total)
        falls.append(n_falls)
        if (episode+1) % 500 == 0:
            metrics = evaluate(agent, epsilon=0.1, episodes=100, seed=10000)
            checkpoints.append({'episode':episode+1, **metrics})
            if verbose:
                print(f'Episode {episode+1:4d} | train return (last 100): {np.mean(rewards[-100:]):7.2f} | validation return: {metrics["mean_return"]:7.2f}', flush=True)
    env.close()
    return agent, rewards, falls, checkpoints

def plot_learning(runs, path):
    fig, axes = plt.subplots(1,2,figsize=(10,3.6))
    for eps,color in zip([0.01,0.1,0.3],['#257c78','#bd7225','#7854a5']):
        group=[r for r in runs if r['epsilon']==eps]
        smoothed=np.array([np.convolve(r['rewards'],np.ones(100)/100,'valid') for r in group])
        x=np.arange(100, smoothed.shape[1]+100)
        axes[0].plot(x, smoothed.mean(0), label=f'epsilon={eps}',color=color)
        axes[0].fill_between(x,smoothed.min(0),smoothed.max(0),alpha=.13,color=color)
        validation=np.array([[c['mean_return'] for c in r['checkpoints']] for r in group])
        axes[1].plot(np.arange(500,3001,500),validation.mean(0), label=f'train epsilon={eps}',color=color)
    axes[0].set(title='Training return: 100-episode average',xlabel='Episode',ylabel='Return')
    axes[1].set(title='Validation with fixed epsilon=0.1',xlabel='Training episode',ylabel='Mean return')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)

def rollout(agent, gif_path, map_path):
    env=make_env(render_mode='rgb_array')
    state,_=env.reset(seed=2026)
    states=[state]
    frames=[Image.fromarray(env.render())]
    total=0
    falls=0
    while True:
        state,reward,terminated,truncated,_=env.step(int(np.argmax(agent.q[state])))
        states.append(state)
        frames.append(Image.fromarray(env.render()))
        total+=reward
        falls+=int(reward == -100)
        if terminated or truncated:
            break
    env.close()
    if not terminated:
        raise RuntimeError('Saved greedy policy did not reach the goal.')
    frames[0].save(gif_path,save_all=True,append_images=frames[1:]+[frames[-1]]*5,duration=250,loop=0)
    fig,ax=plt.subplots(figsize=(9,3.1))
    for row in range(4):
        for col in range(12):
            cliff=row==3 and 1<=col<=10
            ax.add_patch(Rectangle((col-.5,row-.5),1,1,facecolor='#e7a49d' if cliff else '#f3f6f5',edgecolor='white'))
    xy=np.array([(s%12,s//12) for s in states])
    ax.plot(xy[:,0],xy[:,1],'-o',color='#176b63',markersize=4,linewidth=2)
    ax.text(0,3.3,'START',ha='center',fontsize=8)
    ax.text(11,3.3,'GOAL',ha='center',fontsize=8)
    ax.text(5.5,3,'CLIFF: -100, then return to start',ha='center',fontsize=9)
    ax.set(xlim=(-.6,11.6),ylim=(3.6,-.6),xticks=range(12),yticks=range(4),title='Learned greedy route')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(map_path,dpi=160)
    plt.close(fig)
    return {'success':True,'steps':len(states)-1,'return':total,'cliff_falls':falls,'states':states,'seed':2026}
