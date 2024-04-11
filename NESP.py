# Natural Evolution Strategies Algorithm for Reinforcement Learning, by Adam Hung
# Supports parallelization across multiple CPUs
# Sorry if I don't follow coding conventions or my code isn't perfect, I'm not a coder :)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import multiprocessing
import time

class NESPNet(nn.Module):
    def __init__(self, input_features, output_features, h1_nodes, h2_nodes, state_dict = None):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1_nodes)   
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)  
        self.out = nn.Linear(h2_nodes, output_features)  
        if state_dict:
            self.load_state_dict(state_dict)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)   
        return x

class NESP(nn.Module):
    def __init__(self, envs, n_envs = 1, lamb=1, learning_rate = 0.001, stddev = 0.01, h1_nodes = 64, h2_nodes = 64, rseed = None, state_dict_path = None):
        super().__init__()

        # Set random seeds, if desired for reproducibility
        if rseed:
            np.random.seed(rseed)
            torch.manual_seed(rseed)   

        # Gym Env params
        self.envs = envs
        self.n_envs = n_envs
        
        # Initialize Network
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.input_features = envs[0].observation_space.shape[0]
        self.output_features = envs[0].action_space.shape[0]

        # If an existing neural net is supplied, load it into our model. Otherwise, a new one is generated
        if state_dict_path: state_dict = torch.load(state_dict_path)
        else: state_dict = None
        self.net = NESPNet(input_features = self.input_features, 
                         output_features = self.output_features, 
                         h1_nodes = self.h1_nodes, 
                         h2_nodes = self.h2_nodes,
                         state_dict = state_dict)
        # prevents pytorch from performing unnecessary calculations
        for param in self.net.parameters():
            param.requires_grad = False

        # ES params
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.stddev = stddev

        # Progress vars
        self.iterations = 0
        # create a separate network object
        self.best_net = NESPNet(input_features = self.input_features, 
                         output_features = self.output_features, 
                         h1_nodes = self.h1_nodes, 
                         h2_nodes = self.h2_nodes)
        self.best_eval = None

    # Trains model by iterating over generations and updating policy
    def learn(self, iterations = 1, evaluation_episodes = 1, stop_training_at = None):
       
        # again just make sure no gradient calculations are being made
        with torch.no_grad():
            parent = self.net
            eval_time = None
            while self.iterations < iterations:
                s = time.time()
                self.iterations += 1
                # generate new population
                population = self.reproduce(parent)
                # divide it into sections so that each parallel environment can be responsible for its own section
                populations = self.divide_list(population, self.n_envs)

                # evaluate everyone in parallel
                with multiprocessing.Pool(processes=self.n_envs) as pool:
                    eval_lists = pool.map(self.evaluate_multi, [(sub_population, evaluation_episodes, index) for index, sub_population in enumerate(populations)])
                    evaluations = list()
                    # put all the evaluations into one list
                    for i in range(len(eval_lists[0])): 
                        for eval_list in eval_lists:
                            evaluations.append(eval_list[i])

                # check if any members of the population exceeded the best policy we've seen so far
                for index, evl in enumerate(evaluations):
                    if self.best_eval is None or evl > self.best_eval:
                        self.best_eval = evl
                        # we don't have direct access to this policy anymore, so we have to reconstruct it from the noises we stored
                        for name, param in self.net.named_parameters():
                            self.best_net.state_dict()[name].copy_(param + self.noises[int(index/2)][name])
                        print("New best eval: ", self.best_eval/evaluation_episodes)
                        self.save()
                        if self.best_eval/evaluation_episodes >= stop_training_at:
                            return

                print("iteration number: ", self.iterations, "iteration length (s): ", eval_time, ", median reward of current population: " , np.median(evaluations)/evaluation_episodes)
                normalized_evaluations = (evaluations - np.mean(evaluations)) / (np.std(evaluations) + 1e-8)
                # update parameters. param points to actual parameter object, so updates to param affect the real object
                for name, param in parent.named_parameters():   
                    update = torch.zeros_like(param)
                    for index in range(len(population)):
                        # each noise value is used in two policies, see self.get_noises()
                        noise = self.noises[int(index/2)][name]
                        if index%2:
                            noise = -noise
                        # for each data point, increment our weighted sum by (direction perturbed) * (how well this performed)
                        update += noise * normalized_evaluations[index]
                    # scale this by our learning rate and apply it to the parameters to create the parent of the next generation
                    param += self.learning_rate * update

                e =  time.time()
                eval_time = e-s
            for env in self.envs:
                env.close() 
    
    # Facilitates use of Pool object (parallelization)
    def evaluate_multi(self, tuple):
        sub_population = tuple[0]
        evaluation_episodes = tuple[1]
        index = tuple[2]
        evaluations = list()
        worker_env = self.envs[index]
        for child in sub_population:
            evaluations.append(self.evaluate(child, evaluation_episodes, worker_env))
        return evaluations
    
    # Evaluate a policy in the environment by returning the reward gathered over evaluation_episodes
    def evaluate(self, child, evaluation_episodes, worker_env):
        evaluation = 0
        for _ in range(evaluation_episodes):
            state = worker_env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32) 
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                action = child.forward(state)
                state, reward, terminated, truncated, _ = worker_env.step(action.detach().cpu().numpy())
                state = torch.tensor(state, dtype=torch.float32)
                evaluation += reward
        return evaluation
    
    # Returns a new population of policies generated randomly around a parent
    def reproduce(self, parent):
        self.noises = [self.get_noises() for _ in range(int(self.lamb/2))]
        population = []
        for noise in self.noises:
            # for each set of noises, we use the positive values for one child, and the negative values for another
            # this way we need to generate half as many noise values, and we also decrease variance by increasing the symmetry of our noise sampling distribution
            child1_params = {name: param + noise[name] for name, param in parent.named_parameters()}
            child1 = NESPNet(input_features=self.input_features, output_features=self.output_features, 
                            h1_nodes=self.h1_nodes, h2_nodes=self.h2_nodes, state_dict=child1_params)
            child2_params = {name: param - noise[name] for name, param in parent.named_parameters()}
            child2 = NESPNet(input_features=self.input_features, output_features=self.output_features, 
                            h1_nodes=self.h1_nodes, h2_nodes=self.h2_nodes, state_dict=child2_params)
            population.extend([child1, child2])
        return population
    
    # simple method for dividing up a list into as equal sections as possible
    def divide_list(self, list, n_lists_out):
        sublists = [[] for _ in range(n_lists_out)]
        i = 0
        for element in list:
            sublists[i].append(element)
            i = (i+1)%(n_lists_out)
        return sublists
    
    # creates a dictionary of random noise values that exactly matches the shape of our policy network parameter dictionary
    def get_noises(self):
        noises = {
        'fc1.weight': torch.randn((self.h1_nodes, self.input_features), dtype=torch.float32) * self.stddev,
        'fc1.bias': torch.randn(self.h1_nodes, dtype=torch.float32) * self.stddev,
        'fc2.weight': torch.randn((self.h2_nodes, self.h1_nodes), dtype=torch.float32) * self.stddev,
        'fc2.bias': torch.randn(self.h2_nodes, dtype=torch.float32) * self.stddev,
        'out.weight': torch.randn((self.output_features, self.h2_nodes), dtype=torch.float32) * self.stddev,
        'out.bias': torch.randn(self.output_features, dtype=torch.float32) * self.stddev,
        }
        return noises

    def save(self, path = "net.pt"):
        torch.save(self.best_net.state_dict(), path)
    
    # Primarily for visualization of the policy
    def test(self, episodes = 1, path = "net.pt"):

        net = NESPNet(input_features = self.input_features, 
                         output_features = self.output_features, 
                         h1_nodes = self.h1_nodes, 
                         h2_nodes = self.h2_nodes)
        net.load_state_dict(torch.load(path))

        for _ in range(episodes):
            state = self.envs[0].reset()[0]
            state = torch.tensor(state, dtype=torch.float32) 
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                action = net.forward(state)
                state, reward, terminated, truncated, _ = self.envs[0].step(action.detach().cpu().numpy())
                state = torch.tensor(state, dtype=torch.float32)
        
        self.env.close()

