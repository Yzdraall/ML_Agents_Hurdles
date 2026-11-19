# Deep Reinforcement Learning: 110m Hurdles

![Demo of agents running](ML_Agents_Hurdle_Demo.gif)

## Overview

Deep reinforcement learning (PPO, Unity ML-Agents) applied to continuous ragdoll physics: articulated agents learn to keep their balance, run and clear hurdles on a 110 m hurdles track.

- **Two morphologies**: **Walker** (biped, behavior `HurdleWalker`) and **Crawler** (quadruped, behavior `HurdleCrawler`).
- **Emergent locomotion**: no animation clips or scripted gaits; the policies drive joint targets and muscle strengths directly.
- **Parallel training**: 16 walkers and 16 crawlers run simultaneously in one scene, each on its own lane.
- **Inference** with exported `.onnx` networks (Unity Inference Engine).

The demo GIF shows the earlier v1.4 policies, trained for about 100 million steps per behavior.

## Versions

| Component | Version |
|---|---|
| Unity Editor | 6000.0.66f1 |
| `com.unity.ml-agents` | 4.0.2 |
| `com.unity.inputsystem` | 1.17.0 |
| Universal Render Pipeline | 17.0.4 |
| Python | 3.10.12 (supported by mlagents 1.1.0) |
| `mlagents` (pip) | 1.1.0 |

Install PyTorch as described in the [ML-Agents installation docs](https://github.com/Unity-Technologies/ml-agents) **before** installing `mlagents`. The CPU or CUDA build depends on your machine.

## Project structure

```
Assets/
  Scripts/
    HurdleAgentBase.cs       shared agent logic: observations, actions, rewards, curriculum, stats
    WalkerHurdleAgent.cs     biped specifics (13 actuated joints)
    CrawlerHurdleAgent.cs    quadruped specifics (8 actuated joints)
    BodyPartContact.cs       per body-part contact reporting (floor / hurdle / finish)
    GroundContact.cs         foot ground-contact flag
    CameraFollow.cs          follow camera, Tab cycles through agents
  Scenes/HurdlesTraining.unity   training / inference scene (16 walkers + 16 crawlers)
  ML-Agents/                 Walker and Crawler ragdoll prefabs (from the ML-Agents examples)
  Models/Brains/             legacy v1.4 ONNX models (see below)
config/hurdles.yaml          mlagents-learn configuration (both behaviors + curricula)
results/                     training outputs (git-ignored)
```

`Assets/Models/Brains/` contains `HurdleWalker_v1.4_legacy.onnx` and `HurdleCrawler_v1.4_legacy.onnx` and their `.onnx.data` weight files. They were trained with the old observation and action layout, so they **cannot run with the current agents**. They are kept for reference only.

## Training

### 1. Python environment

Windows (PowerShell):

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# install torch first, following the ML-Agents installation docs
pip install mlagents==1.1.0
mlagents-learn --help
```

Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# install torch first, following the ML-Agents installation docs
pip install mlagents==1.1.0
mlagents-learn --help
```

### 2. Run

From the Editor, start the trainer, then press **Play** in Unity when prompted:

```bash
mlagents-learn config/hurdles.yaml --run-id=hurdles_v2
```

From a standalone build (faster, several environments in parallel):

```bash
mlagents-learn config/hurdles.yaml --run-id=hurdles_v2 --env=Build/Hurdles.exe --num-envs=4 --no-graphics
```

To continue an interrupted run, add `--resume`. To monitor training:

```bash
tensorboard --logdir results
```

### Curriculum

`config/hurdles.yaml` defines two independent curricula. Each value is a **fraction of the real hurdle height above the track surface** (`0.0` means no hurdles, `1.0` means real height):

| Lesson | Value | Advance when smoothed mean reward reaches |
|---|---|---|
| Flat | 0.0 | 3.0 |
| Low | 0.3 | 5.0 |
| Medium | 0.7 | 7.0 |
| Real | 1.0 | (final lesson) |

All lessons use `min_lesson_length: 100`. `walker_hurdle_height` is measured on `HurdleWalker`, and `crawler_hurdle_height` on `HurdleCrawler`.

`WalkerHurdleAgent` reads `walker_hurdle_height` and `CrawlerHurdleAgent` reads `crawler_hurdle_height`. If no value is received (for example in inference), the agents use `1.0`, the real height.

## Inference / testing

1. Open `Assets/Scenes/HurdlesTraining.unity`.
2. After training, copy `results/<run-id>/HurdleWalker.onnx` and `results/<run-id>/HurdleCrawler.onnx` to `Assets/Models/Brains/`, together with any `.onnx.data` files.
3. Assign them to the **Model** field of **Behavior Parameters** on the prefabs `Assets/ML-Agents/Walker/Prefabs/Ragdoll/WalkerRagdoll.prefab` and `Assets/ML-Agents/Prefabs/Crawler.prefab`. Assigning at prefab level gives the model to all 16 instances.
4. Press **Play**.

If no model is assigned and no trainer is connected, the agents run the **Heuristic** (manual control).

| Key | Action |
|---|---|
| Tab | Camera: cycle through agents |
| `[` / `]` | Heuristic: select previous / next joint |
| Up / Down | Heuristic: drive the selected joint's X axis |
| Left / Right | Heuristic: drive the selected joint's Y axis |
| Space | Heuristic: flex all joints |

## RL design

### Observations

Vector observations are expressed in a **track-aligned frame**. Its forward axis is the horizontal direction from the start to the nearest point of the lane's `finish` collider, which is straight down the lane.

| Observation | Size |
|---|---|
| Root linear velocity | 3 |
| Root angular velocity | 3 |
| Root forward | 3 |
| Root up | 3 |
| Root height relative to start | 1 |
| Track progress (0 to 1) | 1 |
| Next hurdle: normalised distance | 1 |
| Next hurdle: top height | 1 |
| Per actuated joint: local rotation (quaternion) | 4 |
| Per actuated joint: angular velocity relative to root | 3 |
| Per actuated joint: current strength | 1 |
| Per foot: ground contact | 1 |

| Behavior | Vector observation total |
|---|---|
| HurdleWalker | 16 + 13 x 8 + 2 = **122** |
| HurdleCrawler | 16 + 8 x 8 + 4 = **84** |

In addition, a `RayPerceptionSensor3D` named **HurdleRays** is created at runtime. It sits on a level, track-aligned child object that follows the root.

| Setting | Value |
|---|---|
| Rays | 11 (5 per side, max 45 deg) |
| Ray length | 20 |
| Sphere cast radius | 0.5 |
| Start / end vertical offset | +0.7 / -0.6 |
| Detectable tags | `hurdle`, `floor`, `finish` |
| Excluded layers | agent layers (Walker, Crawler) |
| Observation size | 11 x (3 + 2) = 55 |

### Actions

All actions are continuous, in the range [-1, 1].

- For each actuated joint, there is one target per free axis. X maps to [`lowAngularXLimit`, `highAngularXLimit`], and Y/Z map to plus or minus their limit.
- There is also one strength action per joint, mapped to 0..`maxMuscleForce`.
- Each actuated joint uses a slerp drive in acceleration mode (spring `muscleStrength`, damper `muscleDamper`), so light and heavy body parts respond alike. The agent sets the drive mode and values at initialisation. Joint drive settings in the prefabs or the scene are ignored.

| Behavior | Actuated joints | Actions |
|---|---|---|
| HurdleWalker | 13 | 39 |
| HurdleCrawler | 8 | 20 |

### Rewards

| Term | Value |
|---|---|
| Speed (per step) | `speedRewardScale` (0.01) x `speedMatch` x `lookAtTrack` |
| Effort cost (per step) | -0.002 x mean(action^2) |
| Hurdle row cleared | +0.5 |
| Hurdle hit | -0.5 |
| Finish reached | +10 |
| Fall | -1 |

$$
\text{speedMatch} = \left(1 - \left(\frac{\mathrm{clamp}(\lVert \bar v - v_{target}\,\hat d \rVert,\ 0,\ v_{target})}{v_{target}}\right)^2\right)^2,
\qquad
\text{lookAtTrack} = \frac{\hat h_{root} \cdot \hat d + 1}{2}
$$

Here $\bar v$ is the average body-part velocity, $\hat d$ is the track direction, and $\hat h_{root}$ is the root heading.

### Episode termination

| Condition | Details |
|---|---|
| Finish | A body part touches an object tagged `finish` (collision or trigger), or the root passes the finish line (the near face of the lane's `finish` collider) |
| Fall | Root below start height minus `fallMargin` (Walker 2.75, Crawler 0.5); or dot(root.up, up) < 0.2; or any non-foot body part touches `floor` |
| Timeout | `MaxStep` 5250 physics steps (70 s at 75 Hz), decision period 5 |

### Curriculum and randomisation

- **Lane hurdles**: each agent drives the colliders tagged `hurdle` between its start and its finish line whose span comes within `laneMargin` (2 m) of its lane centre line. A collider belongs to a single agent. Pieces at the same distance along the track form one hurdle row, which moves as a whole and gives one observation and one clearing reward.
- **Hurdle height fraction**: environment parameter (see [Curriculum](#curriculum)). Hurdles are sunk into the track so that their top stays at this fraction of its height above the track surface. A value of 0 disables them.
- **Hurdle position jitter**: ±1 m along the track, per row.
- **Start jitter**: ±0.3 m lateral, ±10 deg yaw.

### Custom TensorBoard metrics

The agents report these through `StatsRecorder`, under each behavior name:

`<Behavior>/Hurdles Cleared`, `<Behavior>/Hurdle Hits`, `<Behavior>/Track Progress`, `<Behavior>/Finish Rate`, `<Behavior>/Finish Time (s)`.

### Physics

Fixed timestep 0.0133 s (75 Hz).

## Results / run history

Earlier runs are stored in `results/`. Suffixes: **M** = multi (walker + crawler trained together), **C** = crawler only, **W** = walker only. All of them used the old behavior names (`walker_config`, `crawler_config`), the old observation/action layout and `time_horizon: 64`, with the same PPO hyperparameters as `config/hurdles.yaml`. No `run_logs/training_status.json` was kept, so final rewards are unknown.

| Run | Behaviors | max_steps | time_horizon | Last checkpoint step | Final reward | Artifacts |
|---|---|---|---|---|---|---|
| v1.3_M | walker_config, crawler_config | 100M / 100M | 64 | n/a | n/a | `configuration.yaml` only |
| v1.4_M | walker_config, crawler_config | 110M / 100M | 64 | walker 100,504,333 / crawler 100,333,859 | n/a | 17 ONNX checkpoints per behavior + final `walker_config.onnx` / `crawler_config.onnx` |
| v1.5_C | crawler_config | 100M | 64 | n/a | n/a | `configuration.yaml` only |
| v1.6_W | walker_config | 20M | 64 | n/a | n/a | `configuration.yaml` only |

The legacy models in `Assets/Models/Brains/` come from v1.4_M.

> TensorBoard curves (reward, Track Progress, Finish Rate, curriculum lessons) will be added to `Docs/images/` after the next training run with `config/hurdles.yaml`.

## Technical stack

- **Engine**: Unity 6 (C#), URP
- **RL**: Unity ML-Agents (PPO)
- **Training / export**: PyTorch, ONNX

## Third-party

![Unity ML-Agents example agents](Docs/images/mlagents_example_agents.png)

The Walker and Crawler ragdoll prefabs and models come from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) example environments. They are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The image above is from the same project.

## License

The project code is released under the MIT License (see [LICENSE](LICENSE)). Third-party assets keep their own licenses (see above).
