## Markov Models Categories
Markov Models can be categorised into four broad classes of models depending upon the autonomy of the system and whether all or part of the information about the system can be observed at each state. 
The Markov Model page at Wikipedia[1] provides a useful matrix that outlines these differences, which will be repeated here:

	Fully Observable 	Partially Observable
Autonomous 	Markov Chain[5] 	Hidden Markov Model[2]
Controlled 	Markov Decision Process[3] 	Partially Observable Markov Decision Process[4]

## HMM Requirements
In order to build a HMM that models the performance of your event's results evaluation you need:

    1. Hidden States
        1. Initial Probability Distribution
        1. Transition Matrix
    1. Sequence of Observations
        1. Observation Likelihood Matrix

The Initial Probability Distribution, along with the Transition Matrix and the Observation Likelihood, make up the parameters of an HMM. 
These are the probabilities you’re figuring out if you have a sequence of observations and hidden states, and attempt to learn which specific HMM could have generated them.


## Finance HMM model

From [HMM-Stock-Market-Prediction](https://github.com/valentinomario/HMM-Stock-Market-Prediction/) research & code repo.

emissions Probability = observation Probability = Hidden State

Initial Model(init_params="st"):

    Started with four hidden states.
    Each state's output was modeled by a Gaussian Mixture Model (GMM) with four components.
    Each observation (n) = Ok:= fracChange, fracHigh,and fracLow
    GMM parameters were initially estimated using k-means clustering and then refined using the fitgmdist function in MATLAB (Expectation-Maximization algorithm). The resulting probability density function served as the initial estimate for the emission matrix.
    Transition probabilities were initially set to a uniform distribution.
    The GMM is fitted on the training dataset: the training algorithm estimates the parameters of each Gaussian component, i.e. the mean vectors and covariance matrices.

Training Data Preparation:

    Used a "rolling window" approach to create training sequences.
    Each sequence consisted of 10 days of observations (latency).
    The window shifted one day at a time to create a series of overlapping sequences.

HMM Training:

    Utilized the hmmtrain function in MATLAB.
    Employed the Baum-Welch algorithm to estimate the model parameters (transition and emission probabilities).
    Initialized the Baum-Welch algorithm with the initial guesses described earlier.


> Pseudocode of 

```text
CLASS GaussianMixtureModel
    PROPERTIES
        means       # List of mean vectors for each Gaussian component
        covariances # List of covariance matrices for each Gaussian component
        weights     # List of weights for each Gaussian component

    METHODS
        INITIALIZE(numComponents, data)
            Initialize means, covariances, and weights randomly or using k-means
        
        EXPECTATION_STEP(data)
            For each data point:
                Compute responsibilities using current parameters
            RETURN responsibilities
        
        MAXIMIZATION_STEP(data, responsibilities)
            Update means, covariances, and weights based on responsibilities
        
        FIT(data, maxIterations, tolerance)
            REPEAT until convergence or maxIterations:
                Call EXPECTATION_STEP
                Call MAXIMIZATION_STEP
                Check for convergence
            RETURN trained parameters

        PREDICT_PROBABILITY(data)
            Compute probability of data points given the trained GMM

END CLASS

---

CLASS HiddenMarkovModel
    PROPERTIES
        numStates           # Number of states in the HMM
        transitionMatrix    # State transition probabilities
        initialProbabilities # Initial state probabilities
        stateModels         # List of GaussianMixtureModel objects for each state

    METHODS
        INITIALIZE(numStates, data)
            Initialize transitionMatrix, initialProbabilities, and stateModels

        FORWARD_ALGORITHM(data)
            Compute forward probabilities using the HMM parameters
            RETURN likelihood of the data

        BACKWARD_ALGORITHM(data)
            Compute backward probabilities using the HMM parameters
            RETURN probabilities for all states at all time steps

        EXPECTATION_STEP(data)
            Use Forward-Backward algorithm to compute expected state occupancies
            RETURN expected state occupancies and state transition counts

        MAXIMIZATION_STEP(data, expectedValues)
            Update transitionMatrix, initialProbabilities, and stateModels

        FIT(data, maxIterations, tolerance)
            REPEAT until convergence or maxIterations:
                Call EXPECTATION_STEP
                Call MAXIMIZATION_STEP
                Check for convergence
            RETURN trained HMM parameters

        PREDICT(data)
            Use Viterbi algorithm to find the most likely state sequence for the data
            RETURN state sequence

END CLASS

---

MAIN PROGRAM
    # Step 1: Load data
    data = Load input data

    # Step 2: Define number of states and GMM components
    numStates = Define number of HMM states
    numComponents = Define number of GMM components per state

    # Step 3: Initialize HMM
    hmm = HiddenMarkovModel()
    hmm.INITIALIZE(numStates, data)

    # Step 4: Train HMM
    maxIterations = Define maximum number of training iterations
    tolerance = Define convergence tolerance
    hmm.FIT(data, maxIterations, tolerance)

    # Step 5: Use HMM for prediction
    predictions = hmm.PREDICT(data)
    % Compute evaluation metrics
    mape = mean(abs((true_values - predicted_values) ./ true_values)) * 100;
    dpa = ?

    # Step 6: Output results
    Print predictions or save to a file

END MAIN PROGRAM
```

> Python HMM

**Gaussian Method**

1. numStates, numComponents = 4
1. latency = 10;         % days aka vectors in sequence;
1. useDynamicEdges = 0
1. totalDiscretizationPoints, discretizationPoints = [50 10 10];    % uniform intervals to discretize observed parameters
1. Freqtrade data params
1. Prediction metrics(mape, dpa)

**Hmmlearn package**

1. n_components (int) – Number of states.
1. n_iter (int, optional) – Maximum number of iterations to perform.