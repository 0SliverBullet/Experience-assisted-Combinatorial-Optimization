# Experience-assisted Combinatorial Optimization
This repository is for SUSTech CS321 Group Project Ⅰ. We are solving *uncapacitated facility location problems* (UFLP) with experience-assisted optimization. 

## Project Details

The main contribution in this project consists of three parts:



1. **Two state-of-the-art algorithms are implemented in Python**: 

- an *enhanced group theory-based optimization algorithm* (EGTOA), from the paper titled "[A fast and efficient discrete evolutionary algorithm for the uncapacitated facility location problem](https://www.sciencedirect.com/science/article/pii/S0957417422019960)".
- an *evolutionary simulated annealing* (ESA) , from the paper titled "[Solving large-scale uncapacitated facility location problems with evolutionary simulated annealing](https://www.tandfonline.com/doi/abs/10.1080/00207540600621003)".

Codes are publicly available on GitHub: Algorithm Implementation.



2. We collect **5 benchmarks for UFLP** from
2. We **propose a new model** *f-c open number* (FCON), a heuristic experience to assist in enhancing existing evolutionary algorithms for better solving UFLP, to predict how many facilities we should open in the optimal solution by using machine learning which trains from small-scale instances and predicts on the large-scale instances. 

![image-20240401143557099](README.assets/image-20240401143557099.png)

![image-20240401143633134](README.assets/image-20240401143633134.png)

FCON can reduce the solution space but not significantly. For the instance with 100 facilities, although FCON claims that the optimal solution exists in the solution with open number $n_{opt} \in [n_{pred}-4, n_{pred}+4]$, the solution space just reduces from $2^{100} (1.3 \times 10^{30})$ to $\sum_{i=n_{pred}-4}^{n_{pred}+4} C(100,i) \in [2.1 \times 10^{12}, 8.0 \times 10^{29}]$, which is not significant effect.

However, I still believe the model is useful to some extent and I will explore more useful heuristic experiences in the future.

(Updated in March 2024)
