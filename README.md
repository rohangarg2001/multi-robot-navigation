Multi-Robot Navigation Framework

This repository provides a framework for simulating multi-robot navigation under different scenarios using 
Simplified MDP and POMDP approaches. Follow the instructions below to run the simulations for each case.

---

## **For Simplified MDP Problem**

1. **Switch to the `mcts_dpw` branch**:
   ```bash
   git checkout mcts_dpw
   ```

2. **Run the simulation**:
   ```bash
   python3 main.py
   ```

---

## **For POMDP Problem**

1. **Switch to the `main` branch**:
   ```bash
   git checkout main
   ```

2. **Run the simulation**:
   ```bash
   python3 main2.py
   ```

---

## **For All Problems**

To modify the number of robots or their goal positions:
1. Open the relevant code file:
   - For Simplified MDP: `main.py`
   - For POMDP: `main2.py`

2. Update the parameters directly in the code:
   ```python
   # Example: Change robot count or target goals
   num_robots = 3
   goal_positions = [[7, 5], [5, 7], [2, 7]]
   ```

---

## **Repository Structure**

- **`mcts_dpw` Branch**: Contains the implementation for the Simplified MDP Problem.
- **`main` Branch**: Contains the implementation for the POMDP Problem.
- **Code Files**:
  - `main.py`: Runs the Simplified MDP simulation.
  - `main2.py`: Runs the POMDP simulation.

---

## **Requirements**

- **Python Version**: Python 3.x
- **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Contact**

For questions, issues, or feedback, please contact:

- **Name**: [Alexandre Carlhammar]
- **Email**: [acarlham@stanford.edu]

- 
- **Name**: [Arpit Dwivedi]
- **Email**: [dwivedi7@stanford.edu]

- 
- **Name**: [Rohan Garg]
- **Email**: [rohang73@stanford.edu]

---
