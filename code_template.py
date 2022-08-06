# Code to accompany the paper "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner
# Code template
#
# This code works on tensorflow version 1.14.0 and python version 3.6.3
#
# For later versions of tensorflow there seems to be a massive overhead in the predict function for some reason, and/or it produces mysterious errors.
# If the code doesn't work, make sure you are using these versions of tf and python.
#
# I used keras version 2.3.1, not sure if this is important, but I recommend this just to be safe.

#Otherwise, if this is not an option, modify the simpler code in the *demos* folder
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms as nxa

N = 20   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
DECISIONS = int(N*(N-1)/2)  #The length of the word we are generating. If we are generating a graph, we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.0001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

# The size of the alphabet. In this file we will assume this is 2. There are a 
# few things we need to change when the alphabet size is larger, such as 
# one-hot encoding the input, and using categorical_crossentropy as a loss 
# function.
n_actions = 2

# Leave this at 2*DECISIONS. The input vector will have size 2*DECISIONS, where
# the first DECISIONS letters encode our partial word (with zeros on the 
# positions we haven't considered yet), and the next DECISIONS bits one-hot 
# encode which letter we are considering now. So e.g. [0,1,0,0,   0,0,1,0] 
# means we have the partial word 01 and we are considering the third letter now.
observation_space = 2*DECISIONS 

len_game = DECISIONS 
state_dim = (observation_space,)

INF = 1000000


# Simplest possible model structure, namely a 3-layer perceptron:
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE)) #Adam optimizer also works well, with lower learning rate
print(model.summary())


def calc_score(state):
    """
    Reward function for your problem.
    Input: a 0-1 vector of length DECISIONS. It represents the graph (or other object) you have created.
    Output: the reward/score for your construction. See files in the *demos* folder for examples.    
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            if state[count] == 1:
                G.add_edge(i,j)
            count += 1

    # Must be connected:
    if not (nx.is_connected(G)):
        return -INF

    # Example, compute the diameter of the graph divided by the size:
    return nxa.diameter(G) / G.size()


#### No need to change anything below here. 


def take_action(n, actions, states, games, prob, step, total_score):
    """
    Samples a single action for each of the `n` games using the given action 
    `prob`abilities, and records the result in the actions/games tensors.

    Inputs:
        n: number of games to sample actions for
        actions: a tensor of size `n` x `len_game`
        states: a tensor of size `n` x `observation_space`
        games: a tensor of size `n` x `observation_space` x `len_game`
        prob: a vector of size `n` of action probabilities
        total_score: a vector of size `n` of rewards for each game
    """
    # So we could see huge performance benefits by batching inference here.
    for i in range(n):
        if np.random.rand() < prob[i]:
            action = 1
        else:
            action = 0
        actions[i][step-1] = action
        states[i] = games[i,:,step-1]

        if (action > 0):
            states[i][step-1] = action
        states[i][DECISIONS + step-1] = 0
        if (step < DECISIONS):
            states[i][DECISIONS + step] = 1

        # calculate final score
        terminal = step == DECISIONS
        if terminal:
            total_score[i] = calc_score(states[i])

        # record sessions 
        if not terminal:
            games[i,:,step] = states[i]

    return actions, states,games, total_score, terminal


def sample(agent, n):
    """
    Samples `n` games from the probabilistic policy defined by the `agent`'s
    neural network.

    Inputs:
        agent: a neural network defining the agent's policy
        n: the number of games to sample
    Outputs:
        games: a tensor of size `n` x `observation_space` x `len_game`
        actions: a tensor of size `n` x `len_game`
        total_score: a vector of size `n`
    """
    games =  np.zeros([n, observation_space, len_game], dtype=int)
    actions = np.zeros([n, len_game], dtype = int)
    total_score = np.zeros([n])
    games[:,DECISIONS,0] = 1 # sets initial 1-hot vector for each game

    # Used as a temporary register during game computation.
    states = np.zeros([n, observation_space], dtype = int)

    step = 0
    while (True):
        step += 1
        prob = agent.predict(games[:,:,step-1], batch_size = n_sessions) 
        actions, states,games, total_score, is_terminal = take_action(n_sessions, actions,states,games,prob, step, total_score)
        if is_terminal:
            break

    return games, actions, total_score


def top_percentile(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    This function returns the top `percentile`% of the games and actions in
    the given batch with respect to their rewards.
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    top_states = []
    top_actions = []
    top_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                top_states.append(states_batch[i])
                top_actions.append(actions_batch[i])
                top_rewards.append(rewards_batch[i])
                counter -= 1
    top_states = np.array(top_states, dtype = int)
    top_actions = np.array(top_actions, dtype = int)
    top_rewards = np.array(top_rewards)
    return top_states, top_actions, top_rewards




super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0
myRand = random.randint(0,1000) #used in the filename

for i in range(1000000): #1000000 generations should be plenty
    games, actions, total_score = sample(model, n_sessions)

    states_batch = np.array(games, dtype = int)
    actions_batch = np.array(actions, dtype = int)
    rewards_batch = np.array(total_score)
    states_batch = np.transpose(states_batch,axes=[0,2,1])

    # NOTE(guy): The use of super_ stuff is to keep the best performing samples
    # across all generations, so we can keep learning from them! Although that
    # kinda seems to me like we should be doing more training cycles per batch.
    states_batch = np.append(states_batch, super_states, axis=0)
    if i>0:
        actions_batch = np.append(actions_batch, np.array(super_actions), axis=0)
    rewards_batch = np.append(rewards_batch, super_rewards)

    elite_states, elite_actions, _ = top_percentile(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
    elite_states = elite_states.reshape((elite_states.shape[0]*elite_states.shape[1], elite_states.shape[2])) # flatten first two dims
    elite_actions = elite_actions.reshape((elite_actions.shape[0]*elite_actions.shape[1]))
    model.fit(elite_states, elite_actions) # TODO: run this multiple times?

    super_sessions = top_percentile(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
    super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
    super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)

    super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
    super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
    super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]

    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:])
    mean_best_reward = np.mean(super_rewards)

    print(f"\n{i}. Best individuals: {np.flip(np.sort(super_rewards))}")
    print(f"Mean reward: {mean_all_reward}")

    if (i%20 == 1): #Write all important info to files every 20 iterations
        with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        with open('best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_all_reward)+"\n")
        with open('best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_best_reward)+"\n")
    if (i%200==2): # To create a timeline, like in Figure 3
        with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(super_actions[0]))
            f.write("\n")
