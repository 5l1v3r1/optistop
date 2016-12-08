# Optimal stopping

I want to see if a neural net can figure out the solution to the [optimal stopping](https://en.wikipedia.org/wiki/Optimal_stopping) problem. I will do this in a supervised fashion, optimizing the expected probability of success. Although the training is supervised, the network will be told the winning strategy; rather, the it will only be told what the best choice *would* have been, not how it can win on average.

In the future, I hope to augment this to use reinforcement learning. It seems like a fairly simple application of RL, but I wanted to start off with supervised learning.

# Results

Yes, a neural net can learn optimal stopping, and it can learn it very quickly. After a bit of training, the LSTM got up to 37% success on 50-timestep sequences. Here is how you can train it yourself:

```
$ go get github.com/unixpickle/optistop
$ cd $GOPATH/src/github.com/unixpickle/optistop/train
$ go run *.go
2016/12/08 17:19:53 iter 0: cost=-0.012777 last=0.000000
2016/12/08 17:19:54 iter 4: cost=-0.014435 last=-0.013168
...
2016/12/08 17:20:23 iter 1112: cost=-0.253706 last=-0.221011
2016/12/08 17:20:23 iter 1116: cost=-0.317275 last=-0.253981
^C
Caught interrupt. Ctrl+C again to terminate.
$ go run *.go -batch=1000 -step=0
2016/12/08 17:20:29 iter 0: cost=-0.365676 last=0.000000
```

Note: training can be terminated by pressing ctrl+c exactly once. If you press it twice, the process will be killed without saving.

The final command checks the average success probability. In this case, it is about 36.6%. The optimal success probability is `1/e=0.3678794412`. Since that average was only over 1000 samples, it is safe to assume that the network has learned the optimal solution.
