import numpy as np
from prettytable import PrettyTable
from agents import DOORmax
from environment.oo_mdp import OOTaxi

# setup
n_repetitions = 1
max_episodes = 5000
max_steps = 100
# initialization of agents and environments
agent_names = ['DOORmax']
envs = [OOTaxi()]
agents = [DOORmax(nS=500, nA=6, r_max=20, gamma=0.95, delta=0.01,
                  env_name='gym-Taxi', k=5, num_atts=envs[0].num_atts,
                  eff_types=['assignment', 'addition'],
                  oo_mdp_dict=envs[0].oo_mdp_dict)]  # alpha/epsilon p.33/34 Diuks Diss


statistics = {}
for agent, env, agent_name in zip(agents, envs, agent_names):
	all_step_times = []
	for i_rep in range(n_repetitions):  # repeat agent training n_repetitions times
		print('Start Agent: ', agent_name, ' current repetition: ', i_rep + 1, '/', n_repetitions)
		_, step_times = agent.train(env, max_episodes=max_episodes, max_steps=max_steps, show_intermediate=False)
		print('steps total: {}, avg step time: {}'.format(len(step_times), np.mean(step_times)))
		agent.reset()

		all_step_times.extend(step_times)

	print('steps total:{}, step time:{}, total time:{}'.format(len(all_step_times)/n_repetitions,
															   np.mean(all_step_times),
															   sum(all_step_times)/n_repetitions))
	statistics[agent_name] = {'avg steps total': len(all_step_times)/n_repetitions,
							  'avg step time': np.mean(all_step_times),
							  'avg total time': sum(all_step_times)/n_repetitions}
