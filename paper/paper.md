
---
title: "MetaMARL: A Framework for Mechanism Design in Multi-Agent Reinforcement Systems"

tags:
  - Python
  - reinforcement learning
  - multi-agent systems
  - mechanism design
  - sustainability
  - simulation

authors:
  - name: Nadine M. Abdelghani
    given-names: Nadine M.
    surname: Abdelghani
    affiliation: "1, 2"
  - name: Elise Devoie
    affiliation: "3"
  - name: Benjamin Rosman
    affiliation: "4"
  - name: Christof Brandtner
    affiliation: "5"
  - name: Guillaume Dumas
    affiliation: "1, 2"

affiliations:
  - name: Mila - Quebec AI Institute, Université de Montréal, Canada
    index: 1
  - name: CHUSJ Azrieli Research Center, Université de Montréal, Canada
    index: 2
  - name: Civil Engineering, Queen's University, Kingston, Canada
    index: 3
  - name: University of the Witwatersrand, Johannesburg, South Africa
    index: 4
  - name: EM Lyon Business School, Lyon, France
    index: 5

date: 13 October 2026

bibliography: paper.bib
---

# Summary

Many societal and environmental systems involve multiple adaptive decision-makers whose individual objectives conflict with long-term collective outcomes. Whether that conflict is resolved depends on the rules under which those decision-makers behave [@ostrom1990; @dietz2003]. Examples include common-pool resources such as fisheries, forests, and water systems [@koster2025sustainable], as well as public policies such as taxation and emissions regulation [@zheng2022ai; @koster2022democratic]. Experimenting with new rules to govern these systems requires simultaneously modeling strategic agent behavior at the local level and the regulatory mechanisms where these agents are embedded at the global level. MetaMARL additionally supports the composition of heterogeneous agent behaviors and regulatory mechanisms through reusable agent and mechanism objects. Its residual state representation enables multiple interventions to be combined within shared environment dynamics, providing a foundation for studying both bilevel optimization and more general multi-agent interactions.

`MetaMARL` is an open-source Python framework for mechanism design in
multi-agent reinforcement learning (MARL). The framework supports mechanism
design as a bilevel optimization problem in which an inner optimization trains
adaptive agents using reinforcement learning, while an outer optimization
searches for environment-level mechanisms that optimize system-wide objectives.
It also supports multi-agent experiments without an outer optimization loop.

The framework includes a fishery benchmark and supports the development of
more realistic water-reservoir management environments, while also providing
modular abstractions for environments, mechanism spaces, optimization
algorithms, and experiment management. These components enable researchers
to develop new benchmark environments, implement custom regulatory
mechanisms, and evaluate existing or novel mechanism design algorithms
within a unified experimental framework. Although motivated by renewable
resource management, `MetaMARL` is designed to support a broad class of
adaptive governance problems in which institutional rules themselves
become optimization variables.

# Statement of need

Most reinforcement learning software focuses on learning optimal agent policies
within fixed environments. Libraries such as RLlib [@liang2018rllib] and
Stable-Baselines3 [@raffin2021stable] provide scalable implementations of modern reinforcement learning algorithms but assume that the environment dynamics, reward functions, and governing rules are predefined and fixed.

Many real-world decision problems instead require optimizing the rules of the
environment rather than only the behavior of its participants. Examples include
resource quotas, taxation schemes, fines, enforcement policies, and other
institutional mechanisms that influence the incentives faced by learning agents.
These problems naturally form bilevel optimization problems in which adaptive
agents respond to regulatory interventions while regulators seek mechanisms that
optimize long-term social or ecological objectives.

Developing research software for these settings presents several challenges:

- integrating reinforcement learning with mechanism optimization,
- supporting non-stationary environments induced by adaptive agents,
- separating mechanism definitions from environment implementations,
- enabling reproducible experiments across different optimization algorithms.

More realistic benchmarks are also needed to evaluate these methods beyond
stylized environments, particularly when agents interact through shared
physical resources and face competing objectives.

`MetaMARL` addresses these challenges through a modular architecture that
decouples environments, mechanism spaces, reinforcement learning algorithms, and
outer-loop optimization methods. The framework enables researchers to rapidly
prototype new regulatory mechanisms while reusing existing learning algorithms
and experimental infrastructure.

# State of the field

Several mature reinforcement learning frameworks provide scalable
implementations of modern policy optimization algorithms.
RLlib [@liang2018rllib] offers distributed reinforcement learning and extensive
support for multi-agent training, while Stable-Baselines3
[@raffin2021stable] provides a lightweight interface for many standard
algorithms. PettingZoo [@pettingzoo] standardizes multi-agent environments and
OpenSpiel [@openspiel] provides a broad collection of games for reinforcement
learning and game-theoretic research.

These libraries primarily focus on learning agent policies under fixed
environmental rules. They provide limited support for optimizing the mechanisms
that define those environments.

`MetaMARL` complements rather than replaces existing reinforcement
learning libraries. It builds upon RLlib for scalable policy optimization while
introducing abstractions for parameterized mechanism spaces, regulator
environments, bilevel optimization workflows, and experiment reporting. This
allows researchers to study adaptive governance problems where both policies and
institutional rules evolve together.

Related work in computational mechanism design includes the AI Economist [@zheng2022ai], which investigates the joint adaptation of economic agents and a regulatory planner through two-level reinforcement learning. MetaMARL addresses a complementary software-engineering objective: providing reusable abstractions for agents, mechanisms, environment dynamics, and interchangeable optimization interfaces across different application domains. Rather than prescribing a particular economic simulation or regulatory learning algorithm, the framework emphasizes compositional experiment construction and the systematic evaluation of alternative mechanism designs.

# Software design

The framework follows a modular architecture composed of four primary
components.

**Environment layer.**
Environment classes define system dynamics independently from regulatory
mechanisms, enabling different application domains, from simple games to
realistic resource-management models, to reuse common optimization
infrastructure.

**Mechanism layer.**
Mechanism spaces describe parameterized regulatory policies such as quotas,
fines, harvest restrictions, or other intervention strategies. New mechanisms
can be introduced without modifying optimization algorithms.

**Optimization layer.**
The framework separates inner-loop policy optimization from outer-loop mechanism
optimization. Reinforcement learning algorithms such as PPO and APPO are used
for agent learning through RLlib, while black-box optimization algorithms such
as Evolution Strategies optimize mechanism parameters.

**Experiment infrastructure.**
Configuration objects define complete experiments in a reproducible manner,
including environments, optimization algorithms, reporting backends, and
training settings. This modular design allows researchers to exchange
optimization algorithms or environments with minimal code changes.

Together, these abstractions allow new research problems to be expressed by
combining reusable software components rather than implementing specialized
training pipelines for each application.

**Composable residual mechanisms.**
A central design choice in MetaMARL is to represent the effects of agents and mechanisms as structured residuals over a shared Markov decision process state. Each mechanism independently maps actions and environmental information to state-update contributions. These residuals are composed through vectorized operations: additive quantities are accumulated, while applicable constraints are combined by intersection. This separates the definition of individual interventions from their composition, allowing users to construct complex regulatory environments from reusable mechanisms. The abstraction supports heterogeneous agents, configurable leader–follower interactions, and extensions involving additional learning or optimization algorithms.

# Research impact statement

`MetaMARL` has been developed as research infrastructure for investigating mechanism design in adaptive multi-agent systems. Its common-pool fishery benchmark provides an executable research workflow in which learning agents interact with a renewable resource under configurable regulatory interventions. Researchers can compare mechanism configurations, train agent policies using PPO or APPO, and evaluate the resulting behavior using economic, ecological, and optimization metrics.

The benchmark demonstrates the framework's capacity to connect domain-specific environment dynamics with reusable mechanism and optimization components. The accompanying configuration files, tests, and tutorials provide a reproducible starting point for extending these experiments to new mechanisms, agent populations, and evaluation objectives.

A central objective of MetaMARL is to support the development of increasingly realistic benchmarking environments for studying mechanism design in multi-agent systems, moving beyond the stylized settings commonly used in the field. Such environments provide testbeds for investigating social dilemmas, multi-agent alignment, and AI safety, where collective outcomes emerge from interactions among adaptive agents under shared environmental constraints.

A more realistic freshwater reservoir-management benchmark is being developed to investigate resource-allocation conflicts among heterogeneous agents interacting through shared hydrological dynamics. Longer-term extensions aim to support additional optimization algorithms and dynamically adapting regulatory agents.

Although motivated by renewable resource management, the framework is applicable
to a broader class of problems in which adaptive agents interact under
modifiable institutional rules. Potential application domains include fisheries,
water resource allocation, environmental regulation, transportation systems, and
other socio-ecological systems involving strategic interactions.

By separating reinforcement learning algorithms from mechanism optimization, the
framework enables researchers to investigate new classes of adaptive governance
problems while leveraging existing reinforcement learning ecosystems.

# Acknowledgements

This work was supported in part by the CIFAR Catalyst Fund (CF-0360) under the
project *Cooperative AI for Climate Action Coordination*.

The authors thank collaborators and participants from the CIFAR Catalyst
project, as well as workshop participants in Lausanne (2025) and Toronto
(2026), for valuable discussions that contributed to the development of this
software.

# AI usage disclosure

Generative AI tools, including OpenAI ChatGPT (GPT-6), were used to assist with drafting documentation and code review. The human authors are responsible for the framework's research objectives, architectural decisions, scientific methodology, and final implementation. All AI-assisted materials were reviewed and edited by the authors, and the software was validated using the project's automated testing and reproducible experiment workflows.

# References
