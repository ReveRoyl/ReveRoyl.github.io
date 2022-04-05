---
layout: post
title:  "phenomenon of ‘Blocking’ on predicted reward"
date:   2022-01-22 09:56:06
categories: discussion
tags: Neuroscience
---



# The so-called ‘delta rule’ has been used to describe learning based on predicted reward. Explain, using equations, the phenomenon of ‘Blocking’. 

Blocking is the phenomenon where there will be no association with the new stimulus is formed when the unconditional reflex has been predicted. It generally refers to the inability to express knowledge or skills due to failures in learning or memory. If the unconditional stimulus is already predicted fully by one stimulus, and the addition of a new stimulus does not provide any new important information about the unconditional stimuli, the unconditional stimuli will not activate or support the learning process which is responsible for establishing the new conditional reflex. Explaining the phenomenon of blocking in detail, here first come the delta law and reinforcement learning model:

The delta law model is a learning model that describes changes in the strength of synapse, such as learning the relationship between stimulus and reward, using gradient descent to find the optimal weight vector. The gradient descent method is to solve the minimum value of the function, i.e., the error, along the direction of gradient descent.

Classical conditioning is a basic form of associative learning which is considered an essential component of complex learning. Typically, classical conditioning occurs when a neutral stimulus (conditional stimulus) is paired closely or consecutively temporally with a biological stimulus (unconditional stimulus) eliciting a reflex behaviour that is still unlearned (unconditional response).

Here is a simple stimuli- reward model:

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202201221004405.png)

In this equation, the w is the weight ratio which is learned with the delta rule from the association of stimulus and reward. It is a vector, which means that its direction can be changed as the stimulus changes. This directionality is necessary for the presence of multiple stimuli. The u is the presence or absence of a stimulus, which takes a binary value between 0 and 1, that is, u is 1 if there is a stimulus, and u is 0 if there is no stimulus.

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202201221005012.png)

Here δ is the predictive error for a given stimulus and reward, which can be predicted by the following equation:

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202201221005829.png)

Here the ε defines a learning rate, i.e., how fast shall we update the association between the stimulus and reward, this value is between 0 and 1. If ε is 0, it means there is no learning, thus this stimulus-reward association model is locked, and the association between stimulus and reward is not updated with any subsequent learning. If ε = 1 means that the association between stimulus and reward will change completely after one learning, this learning model is also generally meaningless for prediction. Usually, this learning rate is individually dependent and different in various conditions. Through this equation, we can predict the efficiency of learning. Another problem is that the external noise will interfere with the forecast. For the external noise error, we can introduce a filter coefficient and a series of convolution operations to eliminate it. 

A simple learning equation based on unconditional reflexes can explain the phenomenon of the blocking well:

![img](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202201221004196.png)

Here ε still represents the learning rate. The λ is the maximum binding strength for a given unconditional stimuli. The σv is the sum of the strengths of associations between all conditional and unconditional stimuli. The δv is the change in binding strength of a particular conditional reflex on various trials. According to this equation, the blocking occurs when λ equals σv, i.e., the binding strength obtained by the conditional stimulus paired with the unconditional stimulus reaches the λ value. There are two groups in one experiment to test it. First, a conditional stimulus is paired with an unconditional stimulus. The second conditional stimulus then undergoes compound conditioning with the first and the same unconditional stimulus. If it is in the absence of blocking, the second combination has little regulation. However, if the combination of the first group is not or weakly regulated, there will be a large amount of associative strength the second combination as well as the first combination in the second group. Blocking occurs because the second stimulus loses relevance. In this equation, blocking will result in an ignorance of new information, which means “stop learning”.

All in all, blocking, in a simple word, is an expression of the learning failure or a classical conditional response. The blocking may play an important role in how animals process information in their environment. Because animals are constantly exposed to numerous stimuli, in which case, selectively responding to those stimuli may keep us away from “brain crashes” caused by too much information. Sometimes, ignorance is a blessing.

