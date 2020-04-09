import numpy as np
from prettytable import PrettyTable
from efficient_rl.agents import Rmax, FactoredRmax, QLearning
from efficient_rl.environment.classical_mdp import ClassicalTaxi
from efficient_rl.environment.factored_mdp import FactoredTaxi


max_episodes = 5000
max_steps = 100
n_repetitions = 1

agent_names = ['Rmax', 'Factored Rmax', 'Q Learning', 'Q Learning optimistic initalization']
envs = [ClassicalTaxi(), FactoredTaxi(), ClassicalTaxi(), ClassicalTaxi()]
agents = [Rmax(M=1, num_states=500, num_actions=6, gamma=0.95, r_max=20, delta=0.01,
               env_name='gym-Taxi'),
          FactoredRmax(M=1, num_states_per_var=[5, 5, 5, 4], num_actions=6, gamma=0.95,
                       r_max=20, delta=0.01, DBNs=envs[1].DBNs,
                       factored_mdp_dict=envs[1].factored_mdp_dict, env_name='gym-Taxi'),
          QLearning(num_states=500, num_actions=6, gamma=0.95, alpha=0.1, epsilon=0.6,
                    optimistic_init=False, env_name='gym-Taxi'),  # alpha/epsilon p.33/34 Diuks Diss
          QLearning(num_states=500, num_actions=6, gamma=0.95, alpha=1, epsilon=0, r_max=20,
                    optimistic_init=True, env_name='gym-Taxi')]  # alpha/epsilon p.33/34 Diuks Diss


statistics = {}

index = 1

# for agent, env, agent_name in zip(agents, envs, agent_names):
for agent, env, agent_name in zip([agents[index]], [envs[index]], [agent_names[index]]):
    all_step_times = []
    for i_rep in range(n_repetitions):  # repeat agent training n_repetitions times
        print('Start Agent: ', agent_name, ' current_repetition: ', i_rep + 1, '/', n_repetitions)
        _, step_times = agent.train(env, max_episodes=max_episodes, max_steps=max_steps)
        print('steps total: {}, avg step time: {}'.format(len(step_times), np.mean(step_times)))
        agent.reset()

        all_step_times.extend(step_times)

    statistics[agent_name] = {'avg steps total': len(all_step_times)/n_repetitions,
                              'avg step time': np.mean(all_step_times),
                              'avg total time': sum(all_step_times)/n_repetitions}
print('\n')
table = PrettyTable(['Agent', 'avg steps total', 'avg step time', 'avg total time'])
for name_of_agent, data_agent in statistics.items():
    table.add_row([name_of_agent,
                   data_agent['avg steps total'],
                   np.round(data_agent['avg step time'], 5),
                   np.round(data_agent['avg total time'], 2)])
print(table)
