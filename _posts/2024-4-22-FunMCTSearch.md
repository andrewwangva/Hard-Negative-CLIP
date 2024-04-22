---
layout: distill
title: Using MCTS to Beat FunSearch
description: Instead of using a genetic algorithm, we experiment with incorporating MCTS
date: 2024-4-21
htmlwidgets: true

authors:
  - name: Andrew Wang
    url:
    affiliations:
      name: MIT
---

# Previous Work

Deepmind introduced [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) [1], a paper that provided a unique way to generate programs that solved optimization problems. They were able to find improvements in practice to algorithmic combinatorial problems such as the Cap set problem or online bin packing. The programs generated were largely heuristic based, but due to their complexity, was able to beat out traditional heuristics created. The paper introduced an interesting idea of using LLM’s as a mutator in a genetic algorithm setup. The authors would repeatedly generate new programs by combining previous ones, creating hybrids and increasing variation. Similar ideas were built on top of the idea of using LLMs as a policy from Deepmind including AlphaGeometry. 

Some of the results from FunSearch seemed pretty impressive, beating the state of the art methods in Cap Set and the online bin-packing problem. In this work, we chose to mainly focus on the online bin-packing problem. The problem involves sequentially placing items of varying sizes into bins with a fixed capacity in real-time, aiming to minimize the number of bins used without knowing future items. To follow FunSearch’s format of the problem, we mutate a function `priority` which assigns a priority to each bin given an item that is then sorted into the highest priority bin:

```python
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -(bins - item)
```

Unfortunately, it doesn’t seem that many in the academic community have been working on improving upon this work outside of Deepmind. The results achieved, specifically in the bin-packing problem requires up to scales of 10^6 LLM calls along with multiple runs. We introduce a new method, based around MCTS in finding heuristic programs given an evaluation function that is competitive at a much smaller scale.

# Incorporating MCTS

MCTS has widely been used in AI systems for games. The idea is to use a policy to narrow down moves to search over and then to evaluate the downstream nodes by simulation and a value function. The core of the MCTS algorithm consists of four distinct steps: Selection, Expansion, Simulation, and Backpropagation. These steps are repeated iteratively to build a search tree.

{% include figure.html path="assets/img/MCTS/MCTS_figure.png" class="img-fluid" caption="Figure taken from the AlphaGo paper. Credit: Silver et al. [2]" %}

In this section, we'll briefly describe how we combined LLM outputs with MCTS. We use the LLM as a policy, prompting it with an existing solution and asking it to modify it slightly. We use the [OR dataset](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html) as in FunSearch, we optimize against OR1 and treat OR2-OR4 as our test set.


### 1. Selection
In each iteration, we choose a node to expand by traversing a tree and choosing the child with the largest Upper Confidence Bound (UCB). The UCB for a node is calculated as follows:

$$
UCB(s, a) = U(s, a) + Q(s, a)
$$

$$
U(s, a) = \frac{V(s)}{1 + N}
$$

$$
Q(s, a) = \sum_{s'}^{\text{simulated nodes}} \frac{V(s')}{N}
$$

Where:
- `V(s)` is the value of the program evaluated against our bin packing dataset.
- `N` is the number of times we've visited or simulated a node in the subtree.

This UCB score serves as a way to tradeoff between exploration and exploitation, both increasing the breadth of our tree along with the depth when we find a strong path.

### 2. Expansion
Expansion is performed using the Large Language Model (LLM) as a policy to generate a few candidate actions. We prompt the LLM to take the existing solution from the current node and modify it slightly to improve results.

### 3. Simulation
Simulation involves using a smaller LLM as our fast rollout policy. We go through multiple rounds of modifying our code slightly as a lookahead into what it may look like after a series of modifications. In our experiments, we use CodeLlama-7b as our smaller, weaker model and GPT-3 as our larger, stronger model. We then update our values for the visit number along with our simulation score for each node in the path.


# Results
First we ran an experiment comparing it to simpler search methods. We compare it to methods that repeatedly applies the improvement prompt. We also consider a simple tree where we build out a tree where each node has two children.

<div style="display: flex; justify-content: space-between;">
  {% include figure.html path="assets/img/MCTS/chain.png" class="img-fluid" caption="Self-improvement Chain"%}
  {% include figure.html path="assets/img/MCTS/binary_tree.png" class="img-fluid" caption="Self-improvement Binary Tree"%}
</div>

Below, we show MCTS outpeforms FunSearch when using a comparable amount of LLM calls from the genetic algorithm and is competitive with the best results Deepmind had at a much smaller scale.

| Methods using 1000 LLM Calls                  | OR1   | OR2   | OR3   | OR4   |
|-----------------------------------------------|-------|-------|-------|-------|
| Zero-shot prompting 1000 times                | 51.9  | 107.7 | 212.0 | 420.35|
| 10 x Self-improvement Chain depth 100         | 51.85 | 107.7 | 212.0 | 420.35|
| Binary tree with 1000 nodes                   | 51.8  | 107.7 | 212.0 | 420.3 |
| FunSearch with 1000 API Calls                 | 51.75 | 107.3 | 211.2 | 418.45|
| MCTS with 1000 API Calls                      | **51.6**  | **106.0** | **207.95**| **410.85**|


| Method                                        | OR1   | OR2   | OR3   | OR4   |
|-----------------------------------------------|-------|-------|-------|-------|
| MCTS with 1000 API Calls                      | **51.6**  | 106.0 | 207.95| 410.85|
| Reported FunSearch with 10^6 LLM Calls        | 51.64 | **105.8** | **207.46**| **410.44**|


Our method finds a better solution for the dataset it's optimizing, OR1, but does not generalize as well - an issue that could be due to lack of hyperparameter tuning along with lacking multiple runs as FunSearch was done. We show below how our solution improves with more iterations of MCTS:

{% include figure.html path="assets/img/MCTS/MCTS_Graph.png" class="img-fluid" %}

Here’s the program our method found after 100 iterations of MCTS if you wanted it to test it out yourself:
```python
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    priority_scores = np.zeros_like(bins, dtype=float)

    for i in range(len(bins)):
        if bins[i] >= 2 * item:
            priority_scores[i] = bins[i] * (1 + np.log(bins[i])) / (bins[i] - item + 1.5) + 0.3 * (len(bins) - i) + 0.1 * item
        elif bins[i] >= item:
            priority_scores[i] = bins[i] * (1 + np.log(np.sqrt(item))) / (bins[i] - item + 1) + 0.25 * (len(bins) - i) + 0.1 * item
        else:
            priority_scores[i] = float('-inf')

    return priority_scores
```

# Conclusion and Limitations

Given the limitation in resources, I’d like to scale this method - I believe it won't take much scaling to beat FunSearch completely. Additionally, MCTS is highly parallelizable, so having access to a strong model with batch inference would greatly improve our results.

Additionally, there are many ways to incorporate MCTS: using an actor-critic setup to transform the problem into a two-player game or performing search on the token-level instead. We chose the simplest setup, which still proved to be quite effective.

In a broader scope, I hope more work is done on scaling on the dimension of inference time. As model and dataset sizes increase, an easy way to squeeze improvements in reasoning tasks with Language Model would be to use search. In many areas where we have a fast and accurate evaluation function, using search was the key to gaining strong results. 

### References:
1.Romera-Paredes, B., Barekatain, M., Novikov, A. et al. Mathematical discoveries from program search with large language models. Nature 625, 468–475 (2024).

2.Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).

3.Trinh, T.H., Wu, Y., Le, Q.V. et al. Solving olympiad geometry without human demonstrations. Nature 625, 476–482 (2024).