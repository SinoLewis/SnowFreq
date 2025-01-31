from hmmlearn import hmm
import numpy as np

## Part 1. Generating a HMM with specific parameters and simulating the exam
print("Setup HMM model with parameters")
# init_params are the parameters used to initialize the model for training
# s -> start probability
# t -> transition probabilities
# e -> emission probabilities
model = hmm.CategoricalHMM(n_components=2, random_state=425, init_params='ste')

# initial probabilities
# probability of starting in the Tired state = 0
# probability of starting in the Happy state = 1
initial_distribution = np.array([0.1, 0.9])
model.startprob_ = initial_distribution

print("Step 1. Complete - Defined Initial Distribution")

# transition probabilities
#        tired    happy
# tired   0.4      0.6
# happy   0.2      0.8

transition_distribution = np.array([[0.4, 0.6], [0.2, 0.8]])
model.transmat_ = transition_distribution
print("Step 2. Complete - Defined Transition Matrix")

# observation probabilities
#        Fail    OK      Perfect
# tired   0.3    0.5       0.2
# happy   0.1    0.5       0.4

observation_probability_matrix = np.array([[0.3, 0.5, 0.2], [0.1, 0.5, 0.4]])
model.emissionprob_ = observation_probability_matrix
print("Step 3. Complete - Defined Observation Probability Matrix")

# simulate performing 100,000 trials, i.e., aptitude tests
trials, simulated_states = model.sample(100000)

# Output a sample of the simulated trials
# 0 -> Fail
# 1 -> OK
# 2 -> Perfect
print("\nSample of Simulated Trials - Based on Model Parameters")
print(trials[:10])
print('\n')
print(simulated_states[:10])
## Part 2 - Decoding the hidden state sequence that leads
## to an observation sequence of OK - Fail - Perfect

# split our data into training and test sets (50/50 split)
X_train = trials[:trials.shape[0] // 2]
X_test = trials[trials.shape[0] // 2:]

model.fit(X_train)

# the exam had 3 trials and your dog had the following score: OK, Fail, Perfect (1, 0 , 2)
exam_observations = [[1, 0, 2]]
predicted_states = model.predict(X=[[1, 0, 2]])
print("Predict the Hidden State Transitions that were being the exam scores OK, Fail, Perfect: \n 0 -> Tired , "
      "1 -> Happy")
print(predicted_states)