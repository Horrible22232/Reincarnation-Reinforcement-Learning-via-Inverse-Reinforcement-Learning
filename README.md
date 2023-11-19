# Reincarnation-Reinforcement-Learning-via-Inverse-Reinforcement-Learning (WIP)

## Overview
This Master's thesis evaluates the concept of Reincarnation Reinforcement Learning (RRL) via Inverse Reinforcement Learning (IRL) in a Crafter environment. A significant challenge in Reinforcement Learning (RL) is the computational intensity and lengthy nature of the learning process due to the "Tabula Rasa" approach, where the agent operates without prior knowledge. This issue is largely due to the exploration problem that existing RL algorithms cannot independently resolve.

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
This README format provides a clear summary of the thesis, its methodologies, and results, making it suitable for a GitHub repository. Remember to add actual links to the `CONTRIBUTING.md` and `LICENSE.md` files if they exist in the repository.


Based on: https://github.com/MarcoMeter/neroRL
