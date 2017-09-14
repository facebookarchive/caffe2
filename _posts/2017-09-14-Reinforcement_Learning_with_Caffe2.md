---
title: Reinforcement learning with Caffe2 
layout: post
author: Caffe2 Team
category: blog
---
![](/static/images/pacman-500.png)

Reinforcement learning (RL) is an area of machine learning focused on teaching agents a complex relationship between its action and behavior, and maximizing a reward after a duration in an environment. The agent can be a game avatar, recommender system, notification bot, or variety of other systems that make decisions. The reward could be points in a game, or more engagement on a website. Facebook uses RL in different ways, with one example being when to let page owners know how their pages are performing. 

Today, we are pleased to announce **RL_Caffe2** ([https://github.com/caffe2/reinforcement-learning-models](https://github.com/caffe2/reinforcement-learning-models)), **a set of RL libraries built on the Caffe2 platform**. Sharing an open-source fork of our Caffe2 RL framework allows us to give back to the community and also collaborate with other institutions as RL finds more applications in industry. 

<!--truncate-->

This project, called RL_Caffe2, contains several RL implementations built on Caffe2 and integrated with OpenAI Gym:

1. **DQN**: An implementation of the Deep Q Learning network as described in [https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

2. **SARSA**: This is a simplification of DQN that assumes the input data is *on-policy*: the policy generating the data is updating in real-time.  The advantage of SARSA is that, during training, we do not need to know what actions are possible.  We only need to know the actions taken.

3. **Actor-Critic**: An implementation of the Actor Critic model as described in [https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf)


* Github (RL_Caffe2): [https://github.com/caffe2/reinforcement-learning-models](https://github.com/caffe2/reinforcement-learning-models)
