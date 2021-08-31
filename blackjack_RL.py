# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:35:28 2021
Programme for identifying an OSR of a BTP using First-Visit MC
@author: Ward Bogaerts
"""

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np 
import enum 
import gym 
from gym import error, spaces, utils 
from gym.utils import seeding
import statistics
import time 
from collections import Counter
import seaborn as sns
import matplotlib.patches as mpatches

##############################################################################
# Set up cards and deck functionalities
# based on https://github.com/adithyasolai/Monte-Carlo-Blackjack.git
##############################################################################

#ranks are the values associated to specific cards. In blackjack, everything
#above a 10 has a value of 10, except for the ace, which has a value of either
#1 or 11
ranks = {
    "two"   : 2, 
    "three" : 3,
    "four"  : 4, 
    "five"  : 5, 
    "six"   : 6, 
    "seven" : 7,
    "eight" : 8, 
    "nine"  : 9,
    "ten"   : 10, 
    "jack"  : 10, 
    "queen" : 10, 
    "king"  : 10, 
    "ace"   : (1,11)
    }

#define a class for the suits
class Suit(enum.Enum): 
    spades   = "spades"
    clubs    = "clubs"
    diamonds = "diamonds"
    hearts   = "hearts"
    
#define a class for a card, a deck is made up out of multiple cards (52 of course)
class Card:
    def __init__(self, suit, rank, value):
        self.suit = suit
        self.rank = rank
        self.value = value
        
    def __str__(self):
        return self.rank + " of " + self.suit.value

#define the deck class, which has functionality to shullfe, deal, peek at the 
#top card and to add a card to the bottom of the deck 
class Deck: 
    def __init__(self,num_decks=1):
        self.cards = []
        for i in range(num_decks): 
            for suit in Suit: 
                for rank, value in ranks.items(): 
                    self.cards.append(Card(suit, rank, value))
    
    #deal one card
    def deal(self): 
        return self.cards.pop(0)
    
    #shuffle the deck 
    def shuffle(self): 
        random.shuffle(self.cards)
        
    #add a card to the bottom of the deck 
    def add_to_bottom(self, card): 
        self.cards.append(card)

    #return all cards still in the deck 
    def __str__(self): 
        res = ""; 
        for card in self.cards: 
            res += card.str(card) + "\n"
        return res
    
    #return the amount of cards still in the deck 
    def __len__(self): 
        return len(self.cards)


##############################################################################
# Implement blackjack rules + create functions to evaluate hands of dealer 
# and player 
# based on https://github.com/adithyasolai/Monte-Carlo-Blackjack.git
##############################################################################

def eval_dealer_hand(dealer_hand): 
    number_aces = 0     #number of aces in the hand
    using_ace_one = 0   #value of hand when we use an ace as a value of 1
    
    #determine the value of the hand when we see the aces as 1
    for card in dealer_hand: 
        if card.rank =="ace": 
            number_aces += 1 
            using_ace_one += card.value[0] #value of 1 for the ace
        else: 
            using_ace_one += card.value 
    
    #if the dealer has an ace, see if using 11 as a value would bring the
    #dealer's hand value closer to [17,21]
    if number_aces > 0: 
        num_aces = 0 
        while num_aces < number_aces: 
            #using_ace_eleven: value of dealer's hand when using a value of 11
            #for aces
            using_ace_eleven = using_ace_one + 10 
            
            if using_ace_eleven > 21: 
                return using_ace_one   #dealer would be bust if using 11 for an ace value
            
            elif using_ace_eleven >= 17 and using_ace_eleven <= 21: 
                return using_ace_eleven 
            
            else: 
                #using an ace as 11 does not bring the total to 17 or higher
                #so if we have another ace, we can try using that as 11 as well   
                using_ace_one = using_ace_eleven
                
            num_aces += 1 
            
        return using_ace_one
    
    else: 
        return using_ace_one

  
    
def eval_player_hand(player_hand): 
    number_aces = 0     #number of aces in the hand
    using_ace_one = 0   #value of hand when we use an ace as a value of 1
    
    #determine the value of the hand when we see the aces as 1
    for card in player_hand: 
        if card.rank =="ace": 
            number_aces += 1 
            using_ace_one += card.value[0] #value of 1 for the ace
        else: 
            using_ace_one += card.value
    
    #if the player has an ace, see if using 11 as a value would bring the
    #dealer's hand value closer to [17,21]
    if number_aces > 0: 
        num_aces = 0 
        while num_aces < number_aces: 
            #using_ace_eleven: value of dealer's hand when using a value of 11
            #for aces
            using_ace_eleven = using_ace_one + 10 
            
            if using_ace_eleven > 21: 
                return using_ace_one   #dealer would be bust if using 11 for an ace value
            
            elif using_ace_eleven >= 17 and using_ace_eleven <= 21: 
                return using_ace_eleven 
            
            else: 
                #using an ace as 11 does not bring the total to 17 or higher
                #so if we have another ace, we can try using that as 11 as well   
                using_ace_one = using_ace_eleven
                
            num_aces += 1 
        
        return using_ace_one
    
    else: 
        return using_ace_one


##############################################################################
# define the dealer policy. This is fixed according to some strict casino rules
# where the dealer keeps on hitting until his hand value is larger or equal than
# 17
# based on https://github.com/adithyasolai/Monte-Carlo-Blackjack.git
##############################################################################

def play_dealer_policy(dealer_hand, deck): 
    hand_value = eval_dealer_hand(dealer_hand)
    
    while hand_value < 17: 
        dealer_hand.append(deck.deal())
        hand_value = eval_dealer_hand(dealer_hand)
        
    return hand_value, dealer_hand, deck 


##############################################################################
# Define OpenAI Gym environment 
# based on https://github.com/adithyasolai/Monte-Carlo-Blackjack.git
##############################################################################

STARTING_CAPITAL = 1000
AMOUNT_DECKS = 2

class BlackjackEnv(gym.Env): 
    metadata = {'render.modes': ['human']}
    
    def __init__(self): 
        super(BlackjackEnv, self).__init__()
        
        #initalize deck and hands of dealer and player 
        self.player_hand = []
        self.dealer_hand = []
        self.deck = Deck(AMOUNT_DECKS)
        
        #define the rewards, 100 for winngin, 0 for a tie and -100 for losing 
        self.reward_options = {"win" : 100, "lose" : -100, "tie" : 0}
        
        #define the possible actions: 0 = hit, 1 = stand 
        self.action_space = spaces.Discrete(2)
        
        #define the observation space. It is a tuple where the first element
        #denotes the possible hand values of the player hand for which he has
        #take an action. These values range from 4 (two 2's) up until 20.
        #If the player's hand value is 21, he automaticallyt wins, so no decision has to be made.
        #The second tuple element is the dealer's up card, the value of which 
        #can range from 2 up until 11
        #The observation space also needs to account for all the possible player
        #hand values when going bust. If the agent decides to hit on 21, he can 
        #reach a max value of 31, considering the ace will be counted as 1 when
        #receiving it at a value of 20. This makes the player's possible states
        #[4-31]
        
        self.observation_space = spaces.Tuple((spaces.Discrete(28), spaces.Discrete(10)))
        
        #initialze the 'done' return from 'step' function to be false 
        self.done = False
        
    def _take_action(self, action): 
        if action == 0: #hit 
            self.player_hand.append(self.deck.deal())
        
        #evaluate player's hand value after taking the action 
        self.player_value = eval_player_hand(self.player_hand)
        
    def step(self, action): 
        self._take_action(action)
        #print('action: ' + str(action))
        
        #end the game if it's done => when player hand is >= 21 or player stands
        if action ==1 or self.player_value >= 21: 
            self.done = True
        else: 
            self.done = False
            
        #initialize and calculate reward for this step
        reward = 0
        
        
        if self.done:   #so stand or player value above 21 
            #calculate reward/loss 
            if self.player_value == 21 and self.dealer_value != 21:     #blackjack, automatic win
                reward = self.reward_options["win"]
            elif self.player_value > 21:    #burnt, automatic lose
                reward = self.reward_options["lose"]
            else: 
                #player stands
                #now it's not clear whether the player has won or lost. For this
                #the dealer has to play first in order to be able to compare their hands
                
                #first, check whether the dealer had Blackjack all along 
                if self.dealer_value == 21: 
                    reward = self.reward_options["lose"]
                else: 
                    self.dealer_value, self.dealer_hand, self.deck = play_dealer_policy(self.dealer_hand, self.deck)
                    
                    if (self.dealer_value < self.player_value and self.player_value < 22) or self.dealer_value > 21: 
                        reward = self.reward_options["win"]
                    elif self.dealer_value == 21 or (self.dealer_value > self.player_value and self.dealer_value < 22) : 
                                              # this only gets checked when the player stands. Both people could have blackjack
                        reward = self.reward_options["lose"]
                    elif self.player_value == self.dealer_value: 
                        reward = self.reward_options["tie"]

        self.capital += reward
        
        #determine the next state to return 
        #state is player's hand value -4 to fit in the [0,22] range used to specify
        #the observation space 
        #the other part of the state is the dealer's up card, with a value of [1,10], 
        #subtracted by -1 to fit the [0,9] range in the observation space definition
        
        #index for player hand value 
        obs_index_player = self.player_value - 4 
        
        #index for dealer's upcard 
        obs_index_dealer = eval_dealer_hand([self.upcard]) - 2 
        
        state = np.array([obs_index_player, obs_index_dealer])
        
        #return state, reward and whether the game is done
        return state, reward, self.done, {}
        
 
    #reset game. The player and dealer hands will be put back in the deck and
    #it will be shuffled. After, the player is dealt 2 cards and the dealer's 
    #upcard is shown/dealt. The starting capital of the player is reset to 
    #STARTING_CAPITAL. The function returns an initial state
    def reset(self):
        self.deck.cards += self.player_hand
        self.deck.cards += self.dealer_hand
        self.deck.shuffle()
        
        self.done = False 
        
        self.capital = STARTING_CAPITAL
        
        #deal the player's hand 
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        #deal the dealer's hand 
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]
        
        self.player_value = eval_player_hand(self.player_hand)
        state_index_player = self.player_value - 4    #convert [4-30] range to [0-26] range
        
        self.upcard = self.dealer_hand[0]
        self.dealer_value = eval_dealer_hand(self.dealer_hand)
        state_index_dealer = eval_dealer_hand([self.upcard]) - 2 #convert [2-11] to [0-9]
        
        state = np.array([state_index_player, state_index_dealer])
        
        #determine whether either dealer of player already have blackjack 
        #initialize rewards first 
        reward = 0
        
        if self.player_value == 21 and (state_index_dealer+2) < 10: #dealer has no 10 of Ace as upcard
            reward = env.reward_options["win"]
            self.done = True 

        elif self.dealer_value == 21 and self.player_value == 21: 
            #reward = env.reward_options["tie"]
            self.done = True 
        
        return state, reward, self.done, {}
     
    #render the game. It shows player balance, player hand, player value
    #dealer's upcard and whether the episode is done
    def render(self, mode='human', close=False): 
        player_hand_ranks = []
        for card in self.player_hand: 
            player_hand_ranks.append(card.rank)
            
        dealer_hand_ranks = []
        for card in self.dealer_hand: 
            dealer_hand_ranks.append(card.rank)
        
        
        print('PLayer capital: ' + str(self.capital))
        print('Player hand: ' + str(player_hand_ranks))
        print('Dealer hand: ' + str(dealer_hand_ranks))
        print('Dealer upcard: ' + str(self.upcard.rank))
        print('Done?: ' + str(self.done))
        print('----------------')


##############################################################################
# Initialize Q-value table and policy map 
##############################################################################

def init_mc(env): 
    #initialize policy to be evaluated. The policy is stochastic. In each state
    #a probability to either hit or stick is determined. Initially these 
    #probabilities are set to 50% for each action
    policy_map = [[[0.5 for k in range(env.action_space.n)]for j in range(env.observation_space[1].n)] for l in range(env.observation_space[0].n)]
    
    #initialize the state-action pair value table. This denotes the (discounted)
    #exected rewards when taking action a in state s. Initially, all these values
    #are set to zero 
    Q_table = [[[0 for k in range(env.action_space.n)] for j in range(env.observation_space[1].n)] for l in range(env.observation_space[0].n)]
    
    #initialize returns array for all the state-action pairs. These are all set
    #to 0 as well 
    returns = Q_table
      
    #initialize learning rate. Defines the weight of new state-action values
    #int the Q_table after each episode. Small learning_rate denotes exploratory 
    #behaviour 
    learning_rate = 0.01
        
    #initialize the probability learning rate epsilon. This rate is analogous to
    #the learning rate, but for the probability table. It determines the weight
    #for adjusting the probability that a certain action is to be taken. A high
    #epsilon yields a smaller increase/decrease for the probability corresponding
    #to the state-action pair
    epsilon = 1
        
    #○epsilon_decay denotes by which factor epsilon is decayed. With decreasing
    #epsilon over the episodes, the agent starts to exploit more than explore.
    epsilon_decay = 0.99999
    
        
    #epsilon cannot go below a certain bound in order to still guarantee some
    #exploration at any time 
    epsilon_min = 0.85
        
    #the discount rate determines in what way the actions prior to a received
    #reward (thus in the same episode) contribute to this reward being received.
    #If the discount rate is 1, no attention is given to the previous actions, 
    #except for the last action which invoked the reward
    discount_rate = 0.8
    
    return policy_map, Q_table, returns, learning_rate, epsilon, epsilon_decay, epsilon_min, discount_rate
  


  

###############################################################################
# Construct main loop for first visit MC
###############################################################################
def loop_mc(env, policy_map, Q_table, returns, learning_rate, epsilon, epsilon_decay, epsilon_min, discount_rate, visited_state): 
    #generate episode using the given policy
    episode_results = []
    episode_reward = 0
    state, reward, env.done, info = env.reset()
    episode_reward += reward 
    
    while env.done == False: 
        #print('state: ' + str(state))
        #determine probabilities to take action 0 (hit) or 1 (stand)
        actions_probs = policy_map[state[0]][state[1]]
        #determine action with highest probability 
        best_action = np.argmax(actions_probs)
        #retract its probability 
        prob_best_action = actions_probs[best_action]
        
        if random.uniform(0,1) < prob_best_action: 
            action = best_action 
        else: 
            if best_action == 1: 
                action = 0
            else: 
                action = 1
                
        #take a step in the environment with the determined action 
        next_state, reward, env.done, info = env.step(action)
        visited_state[state[0]][state[1]] += 1
        episode_reward += reward 

        #append the state - action pair and corresponding reward to results of
        #this episode 
        episode_results.append([state, action, reward])
        
        #start from the next state 
        state = next_state 


    #update the Q_table
    
    episode_index = 0
    for episode in episode_results: 
        state = episode[0]
        action = episode[1]
        
        #update reward according to the discount rate if there were multiple
        #actions taken in this episode 
        #total discounted reward for this episode after this state
        tot_reward_episode = 0
        #exponent of discount rate 
        discount_counter = 0
        
        for index_in_episode in range(episode_index, len(episode_results)): 
            reward = episode_results[index_in_episode][2]
            tot_reward_episode += (discount_rate**discount_counter)*reward
            discount_counter += 1    
        
        #update Q-value of state-action pair via learning rate 
        current_Q = Q_table[state[0]][state[1]][action]
        #Q_table[state[0]][state[1]][action] = current_Q + learning_rate*tot_reward_episode
        Q_table[state[0]][state[1]][action] = current_Q + learning_rate*(tot_reward_episode - current_Q)
        
        episode_index += 1
      
    # update the policy map (probabilities of certain action being taken)    
    epsilon = max(epsilon*epsilon_decay, epsilon_min) 
    for state, action, reward in episode_results: 
        Q_values = Q_table[state[0]][state[1]]
        best_action_updated = np.argmax(Q_values)
        
        policy_map[state[0]][state[1]][best_action_updated] += (1-epsilon)
        policy_map[state[0]][state[1]][best_action_updated] = min(1, policy_map[state[0]][state[1]][best_action_updated])
        
        if best_action_updated == 0: 
            policy_map[state[0]][state[1]][1] = 1 - policy_map[state[0]][state[1]][best_action_updated]
        else: 
            policy_map[state[0]][state[1]][0] = 1 - policy_map[state[0]][state[1]][best_action_updated]
            
    return Q_table, policy_map, epsilon, episode_reward, visited_state
        


###############################################################################
# Define the environment and amount of episodes
###############################################################################

env = BlackjackEnv()

NUM_EPISODES = 1000000

###############################################################################
# Define the training loop function
###############################################################################

def train(): 
    start = time.time() 
    total_reward = 0
    
    reward_per_episode = []
    total_rewards = []
    
    visited_state = np.zeros([env.observation_space[0].n, env.observation_space[1].n], dtype=int)
    
    policy_map, Q_table, returns, learning_rate, epsilon, epsilon_decay, epsilon_min, discount_rate = init_mc(env)
    
    for i in range(NUM_EPISODES):
        Q_table, policy_map, epsilon, reward, visited_state = loop_mc(env, policy_map, Q_table,returns, learning_rate, epsilon, epsilon_decay, epsilon_min, discount_rate, visited_state)
        reward_per_episode.append(reward)
        total_reward += reward
        total_rewards.append(total_reward)
        
    avg_reward = total_reward/NUM_EPISODES
    print('Average reward: ' + str(avg_reward))
    
    ###############################################################################
    # Extract optimal policy + Q-values 
    ###############################################################################
    
    #for the policy, we want a table with rows: player_value, columns: dealer's upcard
    #and values: action to take (0 or 1)
    
    # 1) make empty table with all the states in which the player can be without
    #already being bust (hand value from 4-21). Initialize with zeros, meaning a hit action
    optimal_policy = np.zeros((18,10), dtype=int) 
    
    # 2) for al the state-action pairs where the 1 action (hold) has a higher probability
    #change the values in optimal_policy to 1 
    for i in range(optimal_policy.shape[0]): #for all player hand values which aren't bust 
        for j in range(optimal_policy.shape[1]): #for all dealer upcards [2-11]
            if np.argmax(policy_map[i][j]) == 1 : 
                optimal_policy[i][j] = 1
    
    best_policy_binary = np.zeros([env.observation_space[0].n, env.observation_space[1].n], dtype=int)
    
    for k in range(best_policy_binary.shape[0]):
        for l in range(best_policy_binary.shape[1]): 
           best_policy_binary[k][l] = (np.argmax(Q_table[k][l]))
    
    stop = time.time() 
    training_time = stop-start 
    
    # undo the misleading zeros in the bottom row of the optimal policy for a single run. 
    # These values are set to zero because the agent never reaches this state. Since 0 
    # means hit, we change the values of the unvisited states (when player has 21 and upcard is low)
    # to 1
    for i in range(optimal_policy.shape[0]): 
        for j in range(optimal_policy.shape[1]): 
            if visited_state[i][j] == 0: 
                optimal_policy[i][j] = 1
       
    return optimal_policy, avg_reward, training_time, visited_state, Q_table, total_rewards


###############################################################################
# Define function for testing optimal policy
###############################################################################
def test(best_policy): 
    test_reward = 0
    
    best_policy_binary = best_policy
    
    for i in range(NUM_EPISODES): 
        state, reward, env.done, info = env.reset()
        test_reward += reward 
        while env.done == False: 
            action = best_policy_binary[state[0]][state[1]]
            
            next_state, reward, env.done, info = env.step(action)
            
            test_reward += reward 
            state = next_state
      
    avg_test_reward = test_reward/NUM_EPISODES
        
    print('Average test reward: ' + str(avg_test_reward))

    return avg_test_reward

###############################################################################
# Do a number of test runs 
###############################################################################
AMOUNT_TEST_RUNS = 50

avg_training_rewards = []
avg_testing_rewards = []

Q_tables = []

optimal_policies = []

action_counter_policies = np.zeros((18,10), dtype=int)

visited_states = []

training_times = []


for run in range(AMOUNT_TEST_RUNS): 
    print('Run: ' + str(run))
    optimal_policy, avg_training_reward, training_time, visited_state, Q_table, total_rewards = train()

    # Store optimal policy and training reward + training time
    optimal_policies.append(optimal_policy)
    
    action_counter_policies += optimal_policy
    
    avg_training_rewards.append(avg_training_reward)
    
    Q_tables.append(Q_table)
    
    training_times.append(training_time)
    
    visited_states.append(visited_state)
 
    avg_test_reward = test(optimal_policy)
    
    # Store average test reward
    avg_testing_rewards.append(avg_test_reward)
 
# Results 
avg_training_reward = statistics.mean(avg_training_rewards)
std_training_reward = statistics.stdev(avg_training_rewards, True)

avg_testing_reward = statistics.mean(avg_testing_rewards)
std_testing_reward = statistics.stdev(avg_testing_rewards, True)

avg_training_time = statistics.mean(training_times)

avg_states_visited = np.mean(np.array(visited_states), axis = 0)

avg_Q_table = np.mean(np.array(Q_tables), axis = 0)



# determine the 'average' optimal policy by choosing the action most frequently
# preferred over all the best policies from the individual training loops
avg_optimal_policy = np.zeros((18,10), dtype=int)


for i in range(action_counter_policies.shape[0]): 
    for j in range(action_counter_policies.shape[1]): 
        action_sum = action_counter_policies[i][j]
        if action_sum > (AMOUNT_TEST_RUNS/2) or avg_states_visited[i][j] == 0   : 
            avg_optimal_policy[i][j] = 1

# Calulate the average reward per episode for each episode so far. This denotes
# the learning of the agent towards an asymptotical average reward 
avg_reward_per_episode = []
for i in range(10001): 
    avg_reward_per_episode.append(total_rewards[i]/(i+1))
           

print("Average training reward: " + str(avg_training_reward))
print("Training reward standard deviation: " +str(std_training_reward))
print("Average testing reward: " + str(avg_testing_reward))
print("Testing reward standard deviation: " + str(std_testing_reward)) 
print("Average training time: " + str(avg_training_time))


#%%
###############################################################################
# Plotting 
###############################################################################

# convert average optimal policy to a readable and printable dataframe
avg_optimal_policy_df = pd.DataFrame(avg_optimal_policy, index = range(4,22), columns = range(2,12))

#convert optimal policy of a single training run to a readable and printable dataframe
optimal_policy_df = pd.DataFrame(optimal_policy, index = range(4,22), columns = range(2,12))

fig1 = plt.figure()
ax1 = sns.heatmap(avg_optimal_policy_df, annot=True)
ax1.xaxis.tick_top()
fig1.axes[1].set_visible(False)
plt.xlabel("Dealer's upcard")
plt.ylabel("Player hand value")
plt.title("Average optimal policy for " + str(AMOUNT_TEST_RUNS) + ' runs of ' + str(NUM_EPISODES) + ' episodes')
black_patch = mpatches.Patch(facecolor='black',edgecolor='black', label='Hit')
white_patch = mpatches.Patch(facecolor='bisque',edgecolor='black', label='Stand')
plt.legend(handles=[black_patch, white_patch], bbox_to_anchor=(1,1), loc="upper left")

fig2 = plt.figure()
ax1 = sns.heatmap(optimal_policy_df, annot=True)
ax1.xaxis.tick_top()
fig2.axes[1].set_visible(False)
plt.xlabel("Dealer's upcard")
plt.ylabel("Player hand value")
plt.title("Optimal policy for a single run of  " + str(NUM_EPISODES) + ' episodes')
black_patch = mpatches.Patch(facecolor='black',edgecolor='black', label='Hit')
white_patch = mpatches.Patch(facecolor='bisque',edgecolor='black', label='Stand')
plt.legend(handles=[black_patch, white_patch], bbox_to_anchor=(1,1), loc="upper left")

fig3 = plt.figure()
ax = fig3.gca()
plt.plot(range(10001), avg_reward_per_episode, color = 'red')
ax.set_xlim([0,10000])
plt.xlabel('Episode')
plt.ylabel('Average reward per episode [$]')
plt.title('Evolution of average reward per episode for a single training run of ' + str(NUM_EPISODES) + ' episodes')


fig5 = plt.figure() 
ax = fig5.gca()
plt.bar(range(AMOUNT_TEST_RUNS), avg_training_rewards, color = 'red')
ax.xaxis.set_ticks(range(0, AMOUNT_TEST_RUNS, 5))
ax.set_xticklabels(range(0, AMOUNT_TEST_RUNS, 5))
plt.xlabel('Run')
plt.ylabel('Average reward per training run of ' + str(NUM_EPISODES) + ' episodes [$]')
plt.title('Average reward per training run for ' + str(AMOUNT_TEST_RUNS) + ' runs of ' + str(NUM_EPISODES) + ' episodes')


fig6 = plt.figure() 
ax = fig6.gca()
plt.bar(range(AMOUNT_TEST_RUNS), avg_testing_rewards, color = 'green')
ax.xaxis.set_ticks(range(0, AMOUNT_TEST_RUNS, 5))
ax.set_xticklabels(range(0, AMOUNT_TEST_RUNS, 5))
plt.xlabel('Run')
plt.ylabel('Average reward per testing run of ' + str(NUM_EPISODES) + ' episodes [$]')
plt.title('Average reward per testing run for ' + str(AMOUNT_TEST_RUNS) + ' runs of ' + str(NUM_EPISODES) + ' episodes')

plt.show()

#%%
# small simulation to check the reasoning behing the optimal policy found, where
# the agent tends to hit longer when a high upcard is shown
def upcard_simulation():
    counter = 0
    
    hand_values = []
    while counter < 100000: 
        deckk = Deck(2)
        deckk.shuffle()
        dealer_hand = [deckk.deal(), deckk.deal()]
        player_hand = [deckk.deal(), deckk.deal(), deckk.deal()]
        upcard = dealer_hand[0]
        upcard_value = eval_dealer_hand([upcard])
        if upcard_value >= 7:   
            hand_value, dealer_handd, deckk = play_dealer_policy(dealer_hand, deckk)
            hand_values.append(hand_value)
            counter += 1
    bust_percentage = sum(value > 21 for value in hand_values)/len(hand_values) * 100  
    print('Bust percentage: ' + str(sum(value > 21 for value in hand_values)/len(hand_values) * 100))      



# results sensitivity analysis 

#---alpha = 0.01, epsilon = 0.97 
# Average training reward: -5.597926
# Training reward standard deviation: 0.131098415765891
# Average testing reward: -5.193388
# Testing reward standard deviation: 0.12625655713826542
# Average training time: 100.07075120449066

#-----epsilon 0.85, alpha = 0.001
# Average training reward: -6.528608
# Training reward standard deviation: 0.24767086935392876
# Average testing reward: -5.410986
# Testing reward standard deviation: 0.276141261550337
# Average training time: 106.42655657768249

#-----epsilon 0.6
# Average training reward: -6.50563
# Training reward standard deviation: 0.21334167025057707
# Average testing reward: -5.3518799999999995
# Testing reward standard deviation: 0.2684751544769628
# Average training time: 128.81625898361207

#--alpha=0.01, epsilon_min = .085
# Average training reward: -5.600322
# Training reward standard deviation: 0.11632161658829489
# Average testing reward: -5.25932
# Testing reward standard deviation: 0.190620626547354
# Average training time: 109.00474240303039

#---alpha = 0.00001 epsilon = 0.85
# Average training reward: -9.780872
# Training reward standard deviation: 0.10716572906351511
# Average testing reward: -9.3985
# Testing reward standard deviation: 1.0131979954378039
# Average training time: 144.65461072921752

#----alpha = 0.01, epsilon = 0.6
# Average training reward: -5.554736
# Training reward standard deviation: 0.09960589844020978
# Average testing reward: -5.22082
# Testing reward standard deviation: 0.16548791065038235
# Average training time: 108.94393398284912
