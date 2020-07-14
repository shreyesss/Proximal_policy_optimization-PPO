# PPO-implementation

This is a  keras-Tensorflow bases minimilistic implementation of the RL algorithm PPO (Proximal Policy Optimization) on
  
  a.)Atari games  Breakout and Pong 
  b.)Nintendo's SuperMarioBros 
  c.)Classic control Environment LunarLander.

Features:


1.) The atari games use the no-frameskip environment and implement frameskipping manually. All the frame skipping techniquies used by openai have been implemented minimilistically:
    (step function)
    
    a.) non -sticky action : repeating the same action for a set of four frames 
    b. sticky-action : repating the same  action for last three of the four frames while chossing the previous action for the first frame with a probability of 0.25
    c.) the pixel wise maximum is taken for the last two frames to prevent the dissaperance of ball due to flickering
    

2.)No masking:
   (GAE_and_Targetvalues function)
   
   As a substitute to calculating GAE using masking I update the model at the end of each 'Life' in the game or after a fix number of time_steps(Horizon)
 

3.)No Normalization of GAE values:
 
Contrary to other imlementations I found  that normalization stableizes the training but slows it down a lot. hence for games time-independent and scarce rewards its better to not normalize GAE returns.

Hence normalization of GAE values is used in time-dpendent reward environments of LunarLander and SuperMarioBros and its not used in Atari environments of Pong and Breakout


4.)Soft-Update of old network:

The weights of the network providing the old policy undergo soft update with alpha= 0.1


References:

Frame-skipping :

https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

GAE :

https://arxiv.org/pdf/1506.02438.pdf

PPO :

https://arxiv.org/pdf/1707.06347.pdf






