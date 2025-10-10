# CMPM118
SNN Router
Have a pair of models (e.g., ANN and SNN, or large SNN/small SNN, or sparse SNN/dense SNN) and a router is used to activate one or the other. Much like GPT-5.
- run on xylo, measure entropy (higher = bigger model)

Ideas for router implementation:

0. We could use something like avg_activity = x_spike.float().mean().item() to get the average spike count of the input image. Say we turn a static image into a spiking dataset using rate encoding. We can use that function to take a tensor input for one image and turn it into a single number, the average spike count. This average can be used to determine which model to use. A high spike average = use bigger model, low spike average = use smaller model. Q: Would this work with rate encoding?
1. We use Input Entropy (Complexity). If the same neurons fire -> low entropy, if lots of neurons fire in different, irregular ways -> high entropy. We would check how spread out the spikes are for just one input. 
2. Input change over time. We find a way to detect how much the input image is changing over time, and use that to determine which model to use. 
3. We could implement some sort of locking mechanism so we stall routing until event window completes to prevent mid-prediction switches (hold prediction per window) so there is no mid-router switching.
   
