# ArchLock

The complete implementation will be released soon (The core code has been provided.)



## Overview

Deep Neural Network (DNN) models have significantly advanced various domains of machine learning, exhibiting remarkable performance across a wide range of tasks. However, their susceptibility to adversarial exploitation, where attackers repurpose these models for unintended tasks, poses a significant threat to their integrity. Traditional defense mechanisms have primarily focused on safeguarding model parameters, overlooking the potential of architectural-level interventions.

This repository introduces a pioneering approach to model protection that aims to mitigate this vulnerability by inhibiting the transferability of DNN models at the architectural level. Our method leverages a novel Neural Architecture Search (NAS) algorithm. It employs zero-cost proxies and cross-task search strategies to craft model architectures that are highly performant on their intended tasks while being inherently resistant to repurposing for other applications. More details can be found here (https://openreview.net/pdf?id=e2YOVTenU9).



