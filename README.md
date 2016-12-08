# Optimal stopping

I want to see if a neural net can figure out the solution to the [optimal stopping](https://en.wikipedia.org/wiki/Optimal_stopping) problem. I will do this in a supervised fashion, optimizing for the expected log-likelihood of success. However, just because it's supervised does not mean the neural network will be told the winning strategy; rather, the network will simply be told what the best choice *would* have been, not how it can win on average.

In the future, I hope to augment this to use reinforcement learning. That way, it cannot even be said that the network is ever told the optimal choice to make in a given scenario. For now, though, the supervised setting is easier.
