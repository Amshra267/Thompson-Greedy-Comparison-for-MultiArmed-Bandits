# Exploration-Exploitation Dilemma Solving Methods

In ths repo I implemented two techniques for tackling mentioned tradeoff.
Methods Include:-
 - Epsilon Greedy (With different epsilons)
 - Thompson Sampling(also known as posterior sampling)

The reason for choosing these two only is to show the upper and lower bounds as epsilons are a starting point in dealing with these tradeoffs and Thompson Sampling is considered a recent state of the Art in this field.

<p align = "center">
<b>ENV SPECIFICATIONS</b> - A 10 arm testbed is simulated as same demonstrated in Sutton-Barto Book.
<image width = "600" height = "400" src = True_Rewards.png><br/>
True Reward distribution (Here Action-2 is best)
</p>

## Greedy(or Epsilon Greedies)

we used three different epsilons here for testing
i.e:<br/>
 - epsilon = 0 => Greedy Agent<br>
 - epsilon = 0.01 => exploration with 1% probability
 - epsilon = 0.1 => exploration with 10% probability


<p align = "center">
<b>Averaged Over 2500 independent runs with 1500 timesteps</b></br> 
<image width = "500" height = "300" src = Comparisons.png></br>
<b>Comparison Between Epsilons</b></br>
</br>
<image width = "500" height = "300" src = Optimal_Actions.png><br/>
<b>Percentage Actions selected for epsilon = 0.01</b></br>
</br>
</p>

**1st Conclusion** -> epsilon = 0.01 can be considered best as it is increasing but pretty slow and the percentage Optimal Actions for it is Around 80% means it has more to explore.

## Thompson Sampling 
In this, comparison is performed for Thomspon Sampling with epsilons and Greedy Method to see it's superiority.
I am taking help from this [Reference](Thompson_Sampling.pdf) for understanding the algo.
