Public repository for the Paper **Representational Geometry of Social Inference and
Generalization in a Competitive Game**

https://social-intelligence-human-ai.github.io/docs/camready_8.pdf

*Abstract* 
> The use of an internal model to infer and predict others’ mental states and actions, broadly referred to as Theory of Mind (ToM), is a fundamental aspect of human social intelligence. Nevertheless, it remains unknown how these models are used during social interactions, and how they help an agent generalize to new contexts. We investigated a putative neural mechanism of ToM in a recurrent circuit through the lens of an artificial neural network trained with reinforcement learning (RL) to play a competitive matching pennies game against many algorithmic opponents. The network showed near-optimal performance against unseen opponents, indicating that it had acquired the capacity to adapt against new strategies online. Analysis of recurrent states during play against out-of-training-distribution (OOD) opponents in relation to those of withintraining-distribution (WD) opponents revealed two similarity-based mechanisms by which the network might generalize: mapping to a known strategy (template matching) or known opponent category (interpolation). Even when the network’s strategy cannot be explained by template-matching or interpolation, the recurrent activity fell upon the low-dimensional manifold of the WD neural activity, suggesting the contribution of prior experience with WD opponents. Furthermore, these states occupied low-density edges of the WD-manifold, suggesting that the network can extrapolate beyond any learned strategy or category. Our results suggest that a neural implementation for ToM may be a reservoir of learned representations that provide the capacity for generalization via flexible access and reuse of these stored features.


Relevant Scripts for each Figure:
* Figure 1: ```behavioralperformance.py```, ```opponentswitch.py```, ```winstayloseswitch.py```
* Figure 2: ```lineardecoding.py```,```perturb.py```, ```rsa.py```
* Figure 3: ```scatter_distance.py```,```pca.py```,```density.py```

To install the mprnn package, clone the repo, navigate to the directory and run ```pip install -e .```

```environment.yml``` specifies the dependencies for the conda environment used in the project.
