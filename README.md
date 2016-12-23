# Optimal stopping

This experiment sees if a neural net can solve the [optimal stopping](https://en.wikipedia.org/wiki/Optimal_stopping) problem. The network is trained using [Q-learning](https://en.wikipedia.org/wiki/Q-learning). In the optimal stopping problem, the agent sees a list of candidates and is told if each candidate is superior to all the other candidates. The agent wants to select the best candidate, but must choose whether to accept or select a candidate on the spot (it cannot wait to see the remaining candidates).

Before I used Q-learning, I had a [version of optistop](https://github.com/unixpickle/optistop/tree/supervised) that used supervised learning. The supervised version was very fast (it took about 30 seconds to become optimal on 50-timestep scenarios).

# Results

The agent learns the optimal stopping problem after several minutes of training. You can train the agent as follows:

```
$ go get -u -d github.com/unixpickle/optistop/...
$ cd $GOPATH/src/github.com/unixpickle/optistop/train
$ go run *.go
2016/12/23 16:51:12 Explore coefficient: 0.15346672381831583
2016/12/23 16:51:12 round=0 reward=0.040000
...
2016/12/23 15:32:29 round=500 reward=0.100000
^C
```

To stop training, press Ctrl+C exactly once. Pressing it multiple times will terminate without saving. You can resume training by running `go run *.go` again. When I tested this, the model was finished training by round 500; fewer rounds may work as well.

Now we want to evaluate the trained model. We can get a good estimate of the average reward by passing a few extra flags:

```
$ go run *.go -round 1000 -explore 0
2016/12/23 16:48:42 Explore coefficient: 0
2016/12/23 16:48:43 round=0 reward=0.366000
^C
```

From this output, we see that the average reward was 0.366, which is roughly optimal (`1/e` is optimal).

The `-explore 0` flag is necessary to get the reward from following the optimal policy. Without this flag, the command follows a suboptimal exploratory policy. We use `-round 1000` to get an average over 1000 trials (as opposed to 100 which is the default round size).
