# Efficient Reinforcement Learning [![Build Status](https://travis-ci.com/borea17/efficient_rl.svg?token=rFpzsqEK7NXyNhFzhbms&branch=master)](https://travis-ci.com/borea17/efficient_rl)

**[Motivation](https://github.com/borea17/efficient_rl#motivation)** | **[Summary](https://github.com/borea17/efficient_rl#summary)** | **[Results](https://github.com/borea17/efficient_rl#results)** | **[How to use this repository](https://github.com/borea17/efficient_rl#how-to-use-this-repository)**

-------------------------------------------------------------------------------------

This is a Python *reimplementation* for the Taxi domain of 

* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/Thesis.pdf) (C. Diuks Dissertation)
* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/OORL.pdf) (Paper by C. Diuk et al.)

-------------------------------------------------------------------------------------
### Motivation

It is a well known empirical fact in reinforcement learning that
model-based approaches (e.g., <i>R</i><sub><font
size="4">max</font></sub>) are more sample-efficient than model-free
algorithms (e.g., <i>Q-learning</i>). One of the main reasons may be
that model-based learning tackles the exploration-exploitation dilemma
in a smarter way by using the accumulated experience to bui ld an
approximate model of the environment. Furthermore, it has been 
shown that rich state representations such as in a *factored MDP* can
make model-based learning even more sample-efficient. *Factored MDP*s
enable an effective parametrization of transition and reward dynamics
by using *dynamic Bayesian networks* (DBNs) to represent partial
dependency relations between state variables, thereby the environment dynamics
can be learned with less samples. A major downside of these approaches
is that the DBNs need to be provided as prior knowledge which might be
impossible sometimes.

Motivated by human intelligence, Diuk et al. introduce a new
framework *propositional object-oriented MDPs* (OO-MDPs) to model
environments and their dynamics. As it turns out, humans are way more
sample-efficient than state-of-the-art algorithms when playing games 
such as taxi (Diuk actually performed an experiment). Diuk argues that 
humans must use some prior knowledge when playing this game, he
further speculates that this knowledge might come in form of object
represetations, e.g., identifying horizontal lines as *walls* when 
observing that the taxi cannot move through them. 

Diuk et al. provide a learning algorithm for deterministic
OO-MDPs (<i>DOOR</i><sub><font size="4">max</font></sub>) which
outperforms *factored* <i>R</i><sub><font size="4">max</font></sub>.
As prior knowledge <i>DOOR</i><sub><font size="4">max</font></sub>
needs the objects and relations to consider, which seems more natural
as these may also be used throughout different games. Furthermore,
this approach may also be used to inherit human biases. 

-------------------------------------------------------------------------------------

### Summary

This part shall give an overview about the different reimplemented
algorithms. These can be divided into *model-free* and *model-based* approaches.

#### Model-free Approaches

In model-free algorithms the agent learns the optimal action-value
function (or value function or policy) directly from experience
without having an actual model of the environment. Probably the most
famous model-free algorithm is *Q-learning* which also builds the
basis for the (maybe even more famous) [DQN paper](https://arxiv.org/abs/1312.5602).

##### Q-learning

Q-learning aims to approximate the optimal action-value function 
from which the optimal policy can be inferred. In the simplest case, 
a table (*Q-table*) is used as a function approximator. 

The basic idea is to start with a random action-value function and
then iteratively update this function towards the optimal action-value
function. The update comes after each action *a* with the observed
reward *r* and new state *s<sup>'</sup>*, the update rule is very
simple and is derived from Bellman's optimality equation:

|![\displaystyle Q(s,a)\leftarrow (1-\alpha) Q(s,a) + \alpha\left\[r + \gamma \max_{a^{'}} Q(s^{'}, a^{'})\right\]](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20Q(s%2Ca)%5Cleftarrow%20(1-%5Calpha)%20Q(s%2Ca)%20%2B%20%5Calpha%5Cleft%5Br%20%2B%20%5Cgamma%20%5Cmax_%7Ba%5E%7B'%7D%7D%20Q(s%5E%7B'%7D%2C%20a%5E%7B'%7D)%5Cright%5D)|
|:---:|
  
where &alpha; is the learning rate. To allow for exploration,
Q-learning commonly uses *&epsi;-greedy exploration* or the *Greedy
in the Limit with Infinite Exploration* approach (see [David Silver,
p.13
ff](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)).

Diuk uses two variants of Q-learning:
* **Q-learning**: standard Q-learning approach with &epsi;-greedy
  exploration where parameters &alpha;=0.1 and &epsi;=0.6 have been
  found via parameter search.
* **Q-learning with optimistic initialization**: instead of a some
  random initialization of the Q-table a smart initialization to an
  optimistic value (maximum possible value of any state action pair 
  <img
  src="https://render.githubusercontent.com/render/math?math=v_{max}">)
  is used. Thereby unvisited state-action pairs become more like to be
  visited. Here, &alpha; was is to 1 (deterministic environment) and
  &epsi; to 0 (exploration ensured via initialization).

#### Model-based Approaches 

In model-based approaches the agent learns a model of the environment
by accumulating experience. Then, an optimal action-value
function (or value function or policy) is obtained through *planning*. Planning
can be done exactly or approximately. In the experiments, Diuk et al.
use exact planning, more precisely *value iteration*. The difference
between the three algorithms lies in the way they learn the
environment dynamics.

##### R<sub><font size="4">max</font></sub>

R<sub><font size="4">max</font></sub> is a provably efficient
state-of-the-art algorithm to surpass the exploration-exploitation
dilemma through an intuitive approach: R<sub><font
size="4">max</font></sub> divides state-action pairs into *known*
(state-action pairs which have been visited often enough to build an
accurate transition/reward function) and *unknown*. Whenever a state
is *known*, the algorithm uses the empiriical transition and reward
function for planning. In case a state is *unknown*,  R<sub><font
size="4">max</font></sub> assumes a transition to a fictious state
from which maximum reward can be obtained consistently (hence the name) and it uses
that for planning. Therefore, actions which have not been tried out
(often enough) in the actual state will be preferred unless the
*known* action also leads to maximal return. The parameter *M* defines the number of
observations the agent has to see until it considers a
transition/reward to be known, in a deterministic case such as the
Taxi domain, it can be set to 1. R<sub><font size="4">max</font></sub>
is guaranteed to find a near-optimal action-value function in
polynomial time.

###### Learning transition and reward dynamics

The 5x5 Taxi domain has 500 different states:
 - 5 *x* positions for taxi
 - 5 *y* positions for taxi
 - 5 passenger locations (4 designated locations plus *in-taxi*)
 - 4 destinations

In the standard R<sub><font size="4">max</font></sub> approach without
any domain knowledge (except for the maximum possible reward
*R*<sub><font size="4">max</font></sub>, the
number of states *|S|*, the number of actions *|A|*), the states are
simply enumerated and the agent will
not be able to transfer knowledge throughout the domain. E.g.,
assume the agent performs an action *north* at some location on the
grid and learns the state transition (more precisely it would learn
something like *picking action 0 at state 200 results in ending up in
state 220*). Beeing at the same location but with a different *passenger location* or *destination
location* the agent will not be able to predict the outcome of action
*north*. It will take the agent at least 3000 (*|S| &middot; |A|*)
steps until it has fully learned the 5x5 taxi transition dynamics.
Furthermore, the learned transition and reward dynamics are rather
difficult to interpret.
To address this shortcoming, the agent needs a different
representation and some prior knowledge.

##### Factored R<sub><font size="4">max</font></sub>

Factored R<sub><font size="4">max</font></sub> is an adaptations


##### DOOR<sub><font size="4">max</font></sub>


- adaptations to state-of-the-art Rmax (model based) -> provably efficient algorithm to surpass exploration-exploitation dilemma 
- role of state representations (factored Rmax

-------------------------------------------------------------------------------------

### Results

#### Experimental Setup 

The experimental setup is described on p.31 of Diuks Dissertation. It
consists of testing against six probe states and reporting the number
of steps the agent had to take until the optimal policy for these 6
start states was reached. Since there is some randomness in the
trials, each algorithm runs 100 (`n_repetitions`) times and the
results are then averaged. 

#### Differences between Reimplementation and Diuk

It should be noted that Diuk mainly focused on learning the transition model:
> I will focus on learning dynamics and assume the reward function is available as a black box function (p.61 Diss)

In this reimplementation also the reward function is learned.

#### Dissertation Results (see p.49)

- reimplementation results slightly better since gym environment was
  used in which passenger and destination cannot be at the same
  location, i.e., state space is actually smaller (500 vs 400)

<table>
  <tr>
    <th>Domain knowledge</th>
    <th>Algorithm</th>
    <th colspan="2">Diuks Results<br></th>
    <th colspan="3">Reimplementation Results</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td><u><i># Steps</i></u></td>
    <td><u><i>Time/step</i></u></td>
    <td><u><i># Steps</i></u></td>
    <td><u><i>Time/step</i></u></td>
    <td><u><i>Total Time</i></u></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|<br></td>
    <td>Q-learning</td>
    <td align="center"><b>106859</b></td>
    <td align="center">&lt; 1ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td>Q-learning - optimistic <br>initialization</td>
    <td align="center"><b>29350</b></td>
    <td align="center">&lt;1ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td><i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>4151</b></td>
    <td align="center">74ms</td>
    <td align="center">4071</td>
    <td align="center">2.9ms</td>
    <td align="center">11.7s</td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub>, DBN structure</td>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>1676</b></td>
    <td align="center">97.7ms</td>
    <td align="center">1695</td>
    <td align="center">30.3ms</td>
    <td align="center">51.4s</td>
  </tr>
  <tr>
    <td>Objects, relations to consider,<br><i>R</i><sub><font size="4">max</font></sub></td>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>529</b></td>
    <td align="center">48.2ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (non-<br>videogamers)<br></td>
    <td align="center"><b>101</b></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (videogamers)</td>
    <td align="center"><b>48.8</b></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
  </tr>
</table>

#### Paper Results (see p.7)

<table>
  <tr>
    <th></th>
    <th colspan="3">Diuks Result</th>
    <th colspan="3">Reimplementation Results</th>
  </tr>
  <tr>
    <td></td>
    <td><i>Taxi 5x5 </i></td>
    <td><i>Taxi 10x10</i></td>
    <td><i>Ratio</i></td>
    <td><i>Taxi 5x5</i><br></td>
    <td><i>Taxi 10x10</i></td>
    <td><i>Ratio</i></td>
  </tr>
  <tr>
    <td>Number of states</td>
    <td align="center">500</td>
    <td align="center" >7200</td>
    <td align="center">14.40</td>
    <td align="center">500</td>
    <td align="center">7200</td>
    <td align="center">14.40</td>
  </tr>
  <tr>
    <td>Q-learning<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    <td align="center">
      &nbsp;<br>NA<br>&nbsp;
    </td>
    <td align="center">
      &nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>Q-learning - optimistic initialization<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub><br> 
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
     </td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>1676<br>43.59ms</td>
    <td align="center">&nbsp;<br>19866<br>306.71ms</td>
    <td align="center">&nbsp;<br>11.85<br>7.03</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>529<br>13.88ms</td>
    <td align="center">&nbsp;<br>821<br>293.72ms</td>
    <td align="center">&nbsp;<br>1.55<br>21.16</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>




### How to use this repository

In order to use this repositoy, clone it and run the following command in the directoy of the repository
```python
python3 setup.py install
```
To reproduce the results go into `efficient_rl` folder and run 
```python
python3 dissertation_script.py
```
or
```python
python3 paper_script.py
```
Defaultly, each agent runs only once. To increase the number of repetitions change `n_repetitions` in the scripts. 

If you want to use this repository to play a different game, you may want to look at (). Contributions are welcome and if needed, I will provide a more detailed documentation.
