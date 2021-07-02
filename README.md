# Exploration-Exploitation Dilemma Solving Methods

**Medium article for this repo - [HERE](https://amshra267.medium.com/tackling-exploration-exploitation-dilemma-in-k-armed-bandits-598c0329cf88)**

In ths repo I implemented two techniques for tackling mentioned tradeoff.
Methods Include:-
 - Epsilon Greedy (With different epsilons)
 - Thompson Sampling(also known as posterior samplin

The reason for choosing these two only is to show the upper and lower bounds as epsilons are a starting point in dealing with these tradeoffs and Thompson Sampling is considered a recent state of the Art in this field.

<p align = "center">
<b>ENV SPECIFICATIONS</b> - A 10 arm testbed is simulated as same demonstrated in Sutton-Barto Book.
<image width = "600" height = "400" src = "assets/True_Rewards.png"><br/>
True Reward distribution (Here Action-2 is best)
</p>

##  Comparison Greedy(or Epsilon Greedies and TS

we used three different epsilons here for testing
i.e:<br/>
 - epsilon = 0 => Greedy Agent<br>
 - epsilon = 0.01 => exploration with 1% probability
 - epsilon = 0.1 => exploration with 10% probability</br>
  
and TS


<p align = "center">
<b>Averaged Over 2500 independent runs with 1500 timesteps</b></br> 
<image width = "500" height = "300" src = assets/Comparisons.png></br>
<b>Comparison</b></br>
</br>
<image width = "700" height = "300" src = assets/Optimal_Actions.png><br/>
<b>Percentage Actions selected for epsilon = 0.01 and TS</b></br>
</br>
</p>

**Conclusion** -> epsilon = 0.01 can be considered best for eps-greedies as it is increasing but pretty slow and the percentage Optimal Actions for it is Around 80% in later stages, on the other hand Thomsan Sampling shows a significant improvement in these results as it quickly explores and then exploit the optimal one with percentage goes upto almost 100 even very early!!.

In case you want to know more about TS visit this [Reference](assets/Thompson_Sampling.pdf).

## Updates:
 - Added another version of Thompson Sampling known as **Optimistic Bayesian Sampling**. For Info about the paper and Algorithm, read this papaer - [Optimistic Bayesian Sampling for Contextual Bandits](assets/OTS.pdf)