# Reincarnation-Reinforcement-Learning-via-Inverse-Reinforcement-Learning (WIP)

## Overview
This project explores the application of Reincarnation Reinforcement Learning (RRL) via Inverse Reinforcement Learning (IRL) in a Crafter environment. A significant challenge in Reinforcement Learning (RL) is the computational intensity and lengthy nature of the learning process due to the "Tabula Rasa" approach, where the agent operates without prior knowledge. This issue is largely due to the exploration problem that existing RL algorithms cannot independently resolve.

## Reincarnation Reinforcement Learning
RRL aims to mitigate the inefficiencies and high costs of training where the agent starts from scratch each time. It integrates information gathered from previous training iterations using a trained expert.

## Inverse Reinforcement Learning
In the context of RRL, IRL is utilized to define a reward function that encapsulates the knowledge of the expert, potentially shortening the lengthy and costly exploration phase.

## Methodology
- **Expert Reward Functions:** Four different expert reward functions are defined and evaluated for efficiency.
- **Comparison with Modified Prierarchy:** These functions are compared with a modified version of Prierarchy, which skips the initial imitation learning phase.
- **Hyperparameters:** The choice of hyperparameters, such as the discount factor and the weighting of the expert reward, is crucial for the agent's performance.

## Results
The results indicate that each expert reward function makes learning more efficient. However, despite these findings, the expert reward method is shown to be inferior to the Prierarchy method. Nevertheless, this approach offers significant potential for future research in this area.

## Future Work
This work opens avenues for further exploration in integrating expert knowledge in RL and refining the balance of hyperparameters to optimize agent performance.

## Contributing
Contributions to this research are welcome. Please read [CONTRIBUTING.md](/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the [MIT License](/LICENSE.md) - see the LICENSE.md file for details.

## Acknowledgements
Special thanks to the advisors and peers who supported this research.

## References
This work builds upon and extends several key projects and frameworks in the field:

1. **NeroRL:** A PyTorch-based research framework specializing in Memory-based Deep Reinforcement Learning. [GitHub - neroRL](https://github.com/MarcoMeter/neroRL)
   - Reference: Marco Pleines and Matthias Pallasch.
   
2. **Mastering Diverse Domains through World Models (DreamerV3):** A state-of-the-art, general, and scalable algorithm that leverages world models to achieve superior performance across a broad spectrum of domains . [GitHub - DreamerV3](https://github.com/danijar/dreamerv3)
   - Citation: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* arXiv preprint arXiv:2301.04104.

3. **Crafter:** A benchmark for assessing a spectrum of agent capabilities. [GitHub - Crafter](https://github.com/danijar/crafter)
   - Citation: Hafner, D. (2021). *Benchmarking the Spectrum of Agent Capabilities.* arXiv preprint arXiv:2109.06780.
