# Deep Reinforcement Learning: 110m Hurdles

![Demo of agents running](ML_Agents_Hurdle_Demo.gif)

## Overview

Deep reinforcement learning (**PPO**, **Unity ML-Agents 4.0.2**) applied to continuous 3D ragdoll physics: articulated agents learn from scratch to balance, sprint in their lane, and clear progressive hurdles without any animation clips or scripted gaits.

- **Two morphologies**: **Walker** (biped, behavior `HurdleWalker`) and **Crawler** (quadruped, behavior `HurdleCrawler`).
- **Emergent locomotion**: policies directly actuate multi-axis joint target rotations and muscle strengths (`SlerpDrive` in acceleration mode).
- **Parallel training**: 16 walkers (`3.5x` uniform scale) and 16 crawlers (`2.0x` uniform scale) train simultaneously across 16 multi-lane tracks, with Burst-batched ray perception.
- **Broadcast camera & HUD**: live race-leader tracking, smoothed look-at damping, speed (`km/h`), chrono, and hurdle counters for Editor testing and video capture.
- **Inference** via exported `.onnx` networks (Unity Inference Engine).

---

## Versions

| Component | Version |
|---|---|
| Unity Editor | `6000.0.66f1` |
| `com.unity.ml-agents` | `4.0.2` |
| `com.unity.inputsystem` | `1.17.0` |
| Universal Render Pipeline (URP) | `17.0.4` |
| Python | `3.10.12` |
| `mlagents` (pip) | `1.1.0` |

Install PyTorch (`CUDA` or `CPU`) as described in the [ML-Agents installation docs](https://github.com/Unity-Technologies/ml-agents) **before** installing `mlagents==1.1.0`.

---

## Project Structure

```text
Assets/
  Scripts/
    HurdleAgentBase.cs       shared RL agent: observations, actions, progress rewards, anti-stall, curriculum, stats
    WalkerHurdleAgent.cs     biped specifics (13 actuated joints, shin/foot floor rules)
    CrawlerHurdleAgent.cs    quadruped specifics (8 actuated joints, quadruped hurdle scaling)
    BodyPartContact.cs       per-rigidbody collision/trigger forwarding (floor / hurdle / finish)
    GroundContact.cs         foot ground-contact state (Enter + Stay)
    CameraFollow.cs          damped follow camera, race-leader tracking, and live Broadcast HUD
  Scenes/HurdlesTraining.unity   parallel training & showcase scene (16 walkers + 16 crawlers)
  ML-Agents/                 WalkerRagdoll and Crawler prefabs
  Models/Brains/             exported ONNX brains (LFS-tracked; PyTorch .pt files are git-ignored)
config/hurdles.yaml          unified PPO + curriculum configuration
Docs/images/                 architecture & TensorBoard training curves
results/                     mlagents-learn training outputs (git-ignored)
```

---

## Training

### 1. Python Environment

Windows (PowerShell):

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install torch (CUDA recommended) first, then:
pip install mlagents==1.1.0
mlagents-learn --help
```

Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlagents==1.1.0
```

### 2. Launch Training

**Recommended — Headless Standalone Build (4x–8x faster than Editor):**

1. In Unity: **File -> Build Profiles -> Windows -> Build** into `Build/Hurdles.exe` (with `Assets/Scenes/HurdlesTraining.unity` selected).
2. Run multi-environment PPO training:

```bash
mlagents-learn config/hurdles.yaml --run-id=hurdles_v3 --env=Build/Hurdles.exe --num-envs=4 --no-graphics --time-scale=20
```

**Editor Training (single environment, useful for debugging):**

```bash
mlagents-learn config/hurdles.yaml --run-id=hurdles_v3_debug
```

Resume an interrupted run with `--resume`, and monitor metrics live with:

```bash
tensorboard --logdir results
```

### 3. Curriculum (`config/hurdles.yaml`)

Each behavior advances independently through 4 lessons based on smoothed mean episode reward. Values represent the **fraction of full hurdle height above the track surface** (`0.0` disables hurdles for flat sprint acquisition; `Crawler` caps at `0.55` to match quadruped leg height):

| Lesson | `walker_hurdle_height` | `crawler_hurdle_height` | Reward Threshold to Advance |
|---|---|---|---|
| **0 — Flat** | `0.00` (flat sprint) | `0.00` (flat sprint) | `14.0` |
| **1 — Low** | `0.35` (`0.26–0.91 m`) | `0.20` (`0.15–0.52 m`) | `18.0` |
| **2 — Medium** | `0.70` (`0.52–1.82 m`) | `0.40` (`0.30–1.04 m`) | `22.0` |
| **3 — Real** | `1.00` (`0.74–2.60 m`) | `0.55` (`0.41–1.43 m`) | *(final lesson)* |

---

## Inference & Broadcast Camera Controls

1. Open `Assets/Scenes/HurdlesTraining.unity`.
2. Copy `results/<run-id>/HurdleWalker.onnx` and `results/<run-id>/HurdleCrawler.onnx` (plus their `.onnx.data` weight files!) into `Assets/Models/Brains/`.
3. Assign each `.onnx` asset to the **Model** field of **Behavior Parameters** on `Assets/ML-Agents/Walker/Prefabs/Ragdoll/WalkerRagdoll.prefab` and `Assets/ML-Agents/Prefabs/Crawler.prefab`.
4. Press **Play**.

| Key | Action |
|---|---|
| `Tab` | Camera: cycle to next agent |
| `L` | Camera: toggle **Auto-Follow Race Leader** |
| `H` | Camera: toggle **Live Broadcast HUD** (Chrono, Speed `km/h`, Hurdles, Progress `%`) |
| `[` / `]` | Heuristic (no model): select previous / next joint |
| `Up`/`Down` or `W`/`S` | Heuristic: drive selected joint X axis |
| `Left`/`Right` or `A`/`D` | Heuristic: drive selected joint Y axis |
| `Space` | Heuristic: drive oscillating wave across all X joints |

---

## RL Architecture (v3)

### Vector Observations (Track-Aligned Frame)

All directional vectors are rotated into the lane's horizontal track frame (`forward` = straight down the lane toward the nearest point on the `finish` collider):

| Observation | Size | Notes |
|---|---|---|
| Root linear velocity | 3 | Track frame |
| Root angular velocity | 3 | Track frame |
| Root forward & up vectors | 6 | Track frame (`3 + 3`) |
| Root height above track floor | 1 | Normalised by `heightObservationScale` (`5 m`) |
| Track progress | 1 | Clamped in `[0, 1]` |
| Lateral lane offset | 1 | Signed offset normalised by `maxLaneDrift` (`[-1, 1]`) |
| Next hurdle signed distance | 1 | `[-0.2, 1]` — stays active `1.0 m` past hurdle for trailing-leg clearance |
| Next hurdle top height above floor | 1 | Normalised by `heightObservationScale` |
| Per actuated joint: local rotation | 4 | Quaternion |
| Per actuated joint: rel. angular vel. | 3 | Track frame, relative to root |
| Per actuated joint: current strength | 1 | `[0, 1]` |
| Per foot: ground contact | 1 | `1` if touching `floor`, else `0` |

| Behavior | Vector Observations | Ray Sensor (`HurdleRays`, Burst-batched) | Continuous Actions |
|---|---|---|---|
| `HurdleWalker` | `17 + 13 × 8 + 2 =` **123** | `11 rays × 5 = 55` (`hurdle`, `floor`, `finish`) | **39** (13 joints: free axes + strength) |
| `HurdleCrawler` | `17 + 8 × 8 + 4 =` **85** | `11 rays × 5 = 55` (`hurdle`, `floor`, `finish`) | **20** (8 joints: free axes + strength) |

### Reward Shaping & Anti-Exploit Design

| Term | Formula / Value | Purpose |
|---|---|---|
| **Forward Progress** | `+15.0 × (ΔmaxProgress / trackLength) × lookAt × laneAlign` | Strict potential-based progress; **un-farmable** by oscillating or going slow |
| **Sprint Shaping** | `+0.003 × clamp01(v_fwd / 12) × exp(-0.25 v_lat²) × lookAt × laneAlign` | Dense forward sprint signal; never penalises exceeding `12 m/s` |
| **Hurdle Row Cleared** | `+0.5` per active row | Rewards clearing each of the 7 lane hurdle rows |
| **Finish + Speed Bonus** | `+10.0 + 10.0 × (1 - step / MaxStep)` | Strong terminal incentive to finish in minimum time |
| **Hurdle Hit** | `-0.5` (once per hurdle row) | Discourages ploughing through hurdles |
| **Lane Drift Cost** | `-0.002 × (|lat| / maxLaneDrift)²` | Keeps agents centred in their lane |
| **Time Penalty** | `-0.0015` per decision | Discourages stalling |
| **Effort & Smoothness** | `-0.002 × mean(a²) - 0.001 × mean((a_t - a_{t-1})²)` | Prevents high-frequency joint spasms and wasted torque |
| **Fall / Out-of-Lane / Stall** | `-1.0` + `EndEpisode()` | Terminates falls, lane exits (`> 2.2 m`), or camping (`300` physics steps = `4 s` without `+0.25 m` progress) |

### Custom TensorBoard Metrics

Tracked per behavior via `StatsRecorder`:
- `<Behavior>/Hurdles Cleared`
- `<Behavior>/Hurdle Hits`
- `<Behavior>/Track Progress`
- `<Behavior>/Finish Rate`
- `<Behavior>/Stall Rate`
- `<Behavior>/Finish Time (s)`

---

## Training Analysis & Run History

| Run | Architecture | Steps | Highlights & Lessons Learned |
|---|---|---|---|
| **v1.4_M** *(legacy)* | `132`/`81` obs, `time_horizon: 64` | ~100M | First emergent running demo (`ML_Agents_Hurdle_Demo.gif`); world-space observations, no curriculum. |
| **v2 (`test`)** | `122`/`84` obs, `time_horizon: 1000`, 4-stage curriculum | ~61M+ | **Walker** unlocked all 4 lessons (`Real` at ~38M) and reached finish line in **~24.3 s**. **Crawler** exposed a classic *reward-hacking* exploit: sprinted past rows 1–2 (`0.74 m`), then parked in front of row 3 (`1.40 m`, `Track Progress = 0.3674`) until timeout to avoid `-1.5` crash penalty. |
| **v3 *(current)*** | `123`/`85` obs, progress reward, anti-stall, calibrated Row 7 (`2.60 m`) | Ready | Eliminates hurdle camping (`maxStallSteps = 300`), fixes floating `5.31 m` Row 7 walls, adds action smoothness + Burst-batched raycasts. |

### v2 Run Curves (~60.6M Steps)

| Extrinsic Reward (`Walker` vs `Crawler`) | Curriculum Progression |
|---|---|
| ![Extrinsic Reward](Docs/images/tb_v2_extrinsic_reward.png) | ![Curriculum Lessons](Docs/images/tb_v2_curriculum_lessons.png) |

| Crawler *Reward Hacking* at Row 3 (`Progress = 0.3674`) | Walker First Finishes (~24.3 s) |
|---|---|
| ![Crawler Stall](Docs/images/tb_v2_crawler_progress.png) | ![Walker Finish Time](Docs/images/tb_v2_walker_finish.png) |

---

## Third-Party & License

![Unity ML-Agents example agents](Docs/images/mlagents_example_agents.png)

- Walker and Crawler ragdoll prefabs originate from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) example environments (**Apache License 2.0**).
- Project scripts and configurations are released under the **MIT License** ([LICENSE](LICENSE)).
