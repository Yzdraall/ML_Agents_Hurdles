# Deep Reinforcement Learning: 110m Hurdles

![Demo of agents running](ML_Agents_Hurdle_Demo.gif)

## Project Overview
This project explores deep reinforcement learning (Deep RL) applied to continuous physics. The goal: to train articulated agents to maintain their balance, run, and jump over hurdles on a 110-meter track.

## AI Performance
* **Massive Training:** Models stabilized after **100 million steps** via *Headless* execution (multi-environment parallel).

* **Emergent Behavior:** The agents autonomously learned to coordinate their limbs (Ragdoll physics) to overcome obstacles without any predefined animation scripts.

* **Optimized Inference:** Final models integrated via `.onnx` neural networks.

## The Agents
* **Walker Agent (Biped):** Complex learning focused on body balance and impulse for jumping.

* **Crawler Agent (Quadruped):** Optimization of center of gravity and multi-limb propulsion.

## Technical Stack
* **3D Engine:** Unity (C#)
* **Artificial Intelligence:** Unity ML-Agents Toolkit
* **Modeling:** PyTorch / ONNX Export

## How to Test the Project
1. Clone this repository.

2. Open the project with **Unity**.

3. Load the main training scene.

4. Verify that the final `.onnx` models (located in `results/v1.4_M/`) are correctly assigned in the **Model** field of the **Behavior Parameters** component of the agents.

5. Change the **Behavior Type** to `Inference Only`.

6. Press **Play** to observe the inference in real time.
