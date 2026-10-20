"""Run all experiments, or evaluate the included saved Q-table."""
from pathlib import Path
import argparse
import json
import numpy as np
from agent import SarsaAgent
from utils import train, evaluate, plot_learning, rollout

ROOT=Path(__file__).resolve().parent

def run_experiments():
    output=ROOT/'results'
    output.mkdir(exist_ok=True)
    runs=[]
    candidates={}
    for epsilon in [0.01,0.1,0.3]:
        for seed in [30,31,32]:
            print(f'\nTraining epsilon={epsilon}, seed={seed}',flush=True)
            agent,rewards,falls,checkpoints=train(epsilon=epsilon,seed=seed)
            runs.append({'epsilon':epsilon,'seed':seed,'rewards':rewards,'falls':falls,'checkpoints':checkpoints})
            if seed==30:
                candidates[epsilon]=agent
    # Equal evaluation exploration across candidates, averaged over all seeds.
    scores={eps:float(np.mean([r['checkpoints'][-1]['mean_return'] for r in runs if r['epsilon']==eps])) for eps in candidates}
    selected=max(scores,key=scores.get)
    agent=candidates[selected]
    agent.save(output/'q_table.npy')
    summary={'selected_training_epsilon':selected,'selection_scores':scores,
             'greedy':evaluate(agent),'exploratory':evaluate(agent,epsilon=0.1),
             'random':evaluate(agent,random_policy=True),
             'validation':[{'training_epsilon':r['epsilon'],'seed':r['seed'],'final':r['checkpoints'][-1]} for r in runs],
             'demo':rollout(agent,output/'trained_agent.gif',output/'policy_route.png')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'training_history.json').write_text(json.dumps(runs))
    plot_learning(runs,output/'learning_curves.png')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.evaluate:
        agent=SarsaAgent()
        agent.q=np.load(ROOT/'results'/'q_table.npy')
        print(json.dumps({'greedy':evaluate(agent),'exploratory':evaluate(agent,epsilon=0.1)},indent=2))
    else:
        run_experiments()
