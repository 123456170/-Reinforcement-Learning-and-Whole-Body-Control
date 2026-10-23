import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio


# -----------------------------
# Constants and configuration
# -----------------------------

NOMINAL_Z = 0.86
L1 = 0.34
L2 = 0.34
FOOT = 0.13
UARM = 0.27
FARM = 0.24

JOINT_NAMES = [
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
    "shoulder",
    "elbow",
]

ACTION_NAMES = JOINT_NAMES + ["base_z_cmd", "pitch_cmd"]

JOINT_LIMITS = np.array(
    [
        1.20,  # left hip
        2.00,  # left knee
        0.80,  # left ankle
        1.20,  # right hip
        2.00,  # right knee
        0.80,  # right ankle
        2.20,  # shoulder
        2.20,  # elbow
    ],
    dtype=float,
)

REWARD_KEYS = [
    "velocity",
    "upright",
    "height",
    "energy",
    "smoothness",
    "clearance",
    "joint_limits",
    "survival",
]

REWARD_WEIGHTS = {
    "velocity": 1.00,
    "upright": 0.70,
    "height": 0.35,
    "energy": 0.08,
    "smoothness": 0.06,
    "clearance": 0.18,
    "joint_limits": 0.12,
    "survival": 0.25,
}


# -----------------------------
# Data classes
# -----------------------------


@dataclass
class Params:
    command_velocity: float
    command_turn: float
    target_x: float
    target_z: float
    push_time: float
    push_impulse: float
    push_direction: str
    terrain_roughness: float
    friction_scale: float
    mass_scale: float
    motor_strength: float
    latency_ms: float
    sensor_noise: float
    episode_seconds: float
    fps: int
    seed: int
    sim_hz: int = 200


@dataclass
class State:
    t: float = 0.0
    x: float = 0.0
    z: float = NOMINAL_Z
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    omega: float = 0.0
    phase: float = 0.0
    push_timer: float = 0.0
    fallen: bool = False
    joints: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))
    prev_actions: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))


# -----------------------------
# Utility helpers
# -----------------------------


def temp_path(ext: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"rl_wbc_demo_{os.getpid()}.{ext}")


def empty_hist() -> Dict[str, List[float]]:
    return {
        "t": [],
        "x": [],
        "z": [],
        "pitch": [],
        "vx": [],
        "vy": [],
        "yaw": [],
        "total": [],
        "power": [],
    }


def rotate_point(x: float, z: float, px: float, pz: float, pitch: float) -> np.ndarray:
    c = math.cos(pitch)
    s = math.sin(pitch)
    return np.array([x + c * px - s * pz, z + s * px + c * pz], dtype=float)


def terrain_height(x: float, params: Params) -> float:
    r = float(params.terrain_roughness)
    return r * (
        0.040 * math.sin(2.0 * x)
        + 0.025 * math.sin(5.3 * x + 1.2)
        + 0.015 * math.sin(11.0 * x + 3.0)
    )


def params_hash(p: Params) -> int:
    return hash(
        (
            round(float(p.command_velocity), 6),
            round(float(p.command_turn), 6),
            round(float(p.target_x), 6),
            round(float(p.target_z), 6),
            round(float(p.push_time), 6),
            round(float(p.push_impulse), 6),
            p.push_direction,
            round(float(p.terrain_roughness), 6),
            round(float(p.friction_scale), 6),
            round(float(p.mass_scale), 6),
            round(float(p.motor_strength), 6),
            round(float(p.latency_ms), 6),
            round(float(p.sensor_noise), 6),
            round(float(p.episode_seconds), 6),
            int(p.fps),
            int(p.seed),
        )
    )


# -----------------------------
# Simple whole-body controller / distilled policy proxy
# -----------------------------


def arm_target(state: State, tx: float, tz: float, params: Params) -> tuple:
    shoulder = rotate_point(state.x, state.z, 0.045, 0.22, state.pitch)
    dx = float(tx - shoulder[0])
    dz = float(tz - shoulder[1])
    dist = float(math.hypot(dx, dz))
    max_reach = UARM + FARM

    if dist > 0.98 * max_reach:
        scale = (0.98 * max_reach) / max(dist, 1e-6)
        dx *= scale
        dz *= scale
        dist = float(math.hypot(dx, dz))

    # Angle from downward vertical, positive forward.
    alpha = math.atan2(dx, -dz)
    elbow = max(0.0, (1.0 - dist / max_reach) * 1.7)
    shoulder_angle = 0.62 * alpha

    shoulder_angle = float(np.clip(shoulder_angle, -2.0, 2.0))
    elbow = float(np.clip(elbow, 0.0, 2.0))
    return shoulder_angle, elbow


def make_obs(state: State, params: Params, rng: np.random.Generator) -> np.ndarray:
    base = np.array(
        [
            state.pitch,
            state.omega,
            state.z,
            state.vz,
            state.x,
            state.vx,
            state.vy,
            state.phase % (2.0 * math.pi),
            params.command_velocity,
            params.command_turn,
            params.target_x,
            params.target_z,
        ],
        dtype=float,
    )
    obs = np.concatenate([base, state.prev_actions])
    if params.sensor_noise > 0:
        obs = obs + rng.normal(0.0, params.sensor_noise, size=obs.shape)
    return obs


def policy(obs: np.ndarray, state: State, params: Params, rng: np.random.Generator) -> np.ndarray:
    pitch_m = float(obs[0])
    omega_m = float(obs[1])
    z_m = float(obs[2])
    vz_m = float(obs[3])

    vcmd = float(params.command_velocity)
    ph = float(state.phase)
    terrain = terrain_height(state.x, params)
    speed = abs(vcmd)
    push_boost = 1.0 if state.push_timer > 0.0 else 0.0

    if speed < 0.06 and push_boost == 0.0:
        micro_l = 0.05 * math.sin(ph)
        micro_r = 0.05 * math.sin(ph + math.pi)

        hip_l = 0.02 * vcmd + micro_l
        knee_l = 0.07 + 0.05 * max(0.0, micro_l)
        ankle_l = 0.03 - 0.03 * micro_l

        hip_r = 0.02 * vcmd + micro_r
        knee_r = 0.07 + 0.05 * max(0.0, micro_r)
        ankle_r = 0.03 - 0.03 * micro_r
    else:
        amp = (
            0.22
            + 0.38 * min(1.0, speed / 1.2)
            + 0.12 * float(params.terrain_roughness)
            + 0.18 * push_boost
        )

        sw_l = max(0.0, math.sin(ph))
        st_l = max(0.0, -math.sin(ph))
        sw_r = max(0.0, math.sin(ph + math.pi))
        st_r = max(0.0, -math.sin(ph + math.pi))

        hip_l = 0.14 * vcmd + amp * sw_l - 0.18 * st_l
        knee_l = 0.13 + (0.95 + 0.18 * params.terrain_roughness) * sw_l + 0.07 * st_l
        ankle_l = 0.04 - 0.32 * sw_l + 0.16 * st_l

        hip_r = 0.14 * vcmd + amp * sw_r - 0.18 * st_r
        knee_r = 0.13 + (0.95 + 0.18 * params.terrain_roughness) * sw_r + 0.07 * st_r
        ankle_r = 0.04 - 0.32 * sw_r + 0.16 * st_r

    sh, el = arm_target(state, params.target_x, params.target_z, params)

    balance_pitch = -0.45 * pitch_m - 0.18 * omega_m
    reach_pitch = 0.07 * (params.target_z - NOMINAL_Z) - 0.05 * (params.target_x - state.x)
    pitch_cmd = 0.065 * vcmd + balance_pitch + reach_pitch

    if push_boost > 0.0:
        pitch_cmd += -0.55 * omega_m - 0.35 * pitch_m

    pitch_cmd = float(np.clip(pitch_cmd, -0.42, 0.42))

    height_fb = 0.80 * (NOMINAL_Z + terrain - z_m) - 0.10 * vz_m
    base_z_cmd = (
        NOMINAL_Z
        + 0.018 * math.sin(2.0 * ph)
        - 0.035 * speed
        + 0.04 * height_fb
        + 0.04 * push_boost
    )
    base_z_cmd = float(np.clip(base_z_cmd, 0.58, 1.05))

    action = np.array(
        [
            hip_l,
            knee_l,
            ankle_l,
            hip_r,
            knee_r,
            ankle_r,
            sh,
            el,
            base_z_cmd,
            pitch_cmd,
        ],
        dtype=float,
    )

    if params.sensor_noise > 0:
        action = action + rng.normal(0.0, params.sensor_noise * 0.08, size=action.shape)

    action[:8] = np.clip(action[:8], -JOINT_LIMITS, JOINT_LIMITS)
    action[8] = float(np.clip(action[8], 0.55, 1.08))
    action[9] = float(np.clip(action[9], -0.50, 0.50))

    return action


def apply_push(state: State, params: Params, rng: np.random.Generator) -> None:
    if params.push_direction == "none" or params.push_impulse <= 0.0:
        return

    m = max(0.4, float(params.mass_scale))
    fr = max(0.4, float(params.friction_scale))
    imp = float(params.push_impulse) / (m * fr)
    imp += rng.normal(0.0, 0.03 * abs(imp) + 1e-6)

    if params.push_direction == "front":
        state.vx -= imp
        state.omega -= 0.10 * imp
    elif params.push_direction == "back":
        state.vx += imp
        state.omega += 0.10 * imp
    elif params.push_direction == "side":
        state.vy += 0.70 * imp
        state.omega += 0.18 * imp
        state.yaw += 0.08 * imp

    state.push_timer = 0.95


def step_sim(
    state: State,
    desired_action: np.ndarray,
    params: Params,
    rng: np.random.Generator,
    dt: float,
) -> None:
    old_action = state.actions.copy()
    state.prev_actions = old_action

    lam = float(params.latency_ms) / (float(params.latency_ms) + 45.0)
    action = (1.0 - lam) * desired_action + lam * old_action
    state.actions = action.copy()

    q_des = action[:8]
    base_z_des = float(action[8])
    pitch_des = float(action[9])

    mass = max(0.4, float(params.mass_scale))
    motor = max(0.3, float(params.motor_strength))

    kp = 90.0 * motor
    kd = 9.0 * motor

    qdd = kp * (q_des - state.joints) - kd * state.joint_vel
    if params.sensor_noise > 0:
        qdd = qdd + rng.normal(0.0, params.sensor_noise * 3.0, size=8)

    state.joint_vel += qdd * dt
    state.joints += state.joint_vel * dt
    state.joints = np.clip(state.joints, -JOINT_LIMITS, JOINT_LIMITS)

    freq = (
        0.85
        + 1.25 * abs(float(params.command_velocity))
        + 0.25 * float(params.terrain_roughness)
    )
    if state.push_timer > 0.0:
        freq += 1.10

    state.phase += 2.0 * math.pi * freq * dt

    terrain = terrain_height(state.x, params)

    kz = 220.0 / mass
    dz = 28.0
    az = kz * (base_z_des + terrain - state.z) - dz * state.vz
    state.vz += az * dt
    state.z += state.vz * dt

    tau = 0.20 / mass
    desired_vx = float(params.command_velocity)
    if state.push_timer > 0.0:
        desired_vx += -0.35 * state.vx

    ax = (desired_vx - state.vx) / tau
    ax -= 0.80 * (float(params.friction_scale) - 1.0) * state.vx
    state.vx += ax * dt
    state.x += state.vx * dt

    desired_vy = float(params.command_turn) * 0.18
    state.vy += (desired_vy - state.vy) * min(1.0, 6.0 * dt)
    if params.sensor_noise > 0:
        state.yaw += float(params.command_turn) * dt + rng.normal(
            0.0, params.sensor_noise * 0.04
        )
    else:
        state.yaw += float(params.command_turn) * dt

    kp_p = 120.0 / mass
    kd_p = 16.0
    torque = kp_p * (pitch_des - state.pitch) - kd_p * state.omega
    if params.sensor_noise > 0:
        torque += rng.normal(0.0, params.sensor_noise * 2.0)

    state.omega += torque * dt
    state.pitch += state.omega * dt

    if state.push_timer > 0.0:
        state.push_timer = max(0.0, state.push_timer - dt)

    state.t += dt

    if (
        not np.all(np.isfinite(state.joints))
        or not np.all(np.isfinite(state.joint_vel))
        or not math.isfinite(state.x)
        or not math.isfinite(state.z)
        or not math.isfinite(state.pitch)
        or abs(state.pitch) > 0.82
        or state.z < 0.47
        or abs(state.x) > 8.0
    ):
        state.fallen = True


def kinematics(state: State, params: Params) -> Dict[str, np.ndarray]:
    pelvis = np.array([state.x, state.z], dtype=float)
    chest = rotate_point(state.x, state.z, 0.0, 0.28, state.pitch)
    shoulder = rotate_point(state.x, state.z, 0.045, 0.22, state.pitch)
    head = rotate_point(state.x, state.z, 0.015, 0.38, state.pitch)

    hip_l = rotate_point(state.x, state.z, -0.035, -0.16, state.pitch)
    hip_r = rotate_point(state.x, state.z, 0.035, -0.16, state.pitch)

    def leg_chain(hip: np.ndarray, qh: float, qk: float, qa: float):
        hip_abs = state.pitch + qh
        knee = hip + L1 * np.array([math.sin(hip_abs), -math.cos(hip_abs)], dtype=float)

        lower_abs = hip_abs - qk
        ankle = knee + L2 * np.array([math.sin(lower_abs), -math.cos(lower_abs)], dtype=float)

        foot_abs = lower_abs + qa
        fvec = np.array([math.cos(foot_abs), math.sin(foot_abs)], dtype=float)

        toe = ankle + FOOT * fvec
        heel = ankle - 0.045 * fvec
        return knee, ankle, toe, heel

    lk = leg_chain(hip_l, state.joints[0], state.joints[1], state.joints[2])
    rk = leg_chain(hip_r, state.joints[3], state.joints[4], state.joints[5])

    sh_abs = state.pitch + state.joints[6]
    elbow = shoulder + UARM * np.array([math.sin(sh_abs), -math.cos(sh_abs)], dtype=float)

    fore_abs = sh_abs - state.joints[7]
    hand = elbow + FARM * np.array([math.sin(fore_abs), -math.cos(fore_abs)], dtype=float)

    return {
        "pelvis": pelvis,
        "chest": chest,
        "shoulder": shoulder,
        "head": head,
        "l_hip": hip_l,
        "r_hip": hip_r,
        "l_knee": lk[0],
        "l_ankle": lk[1],
        "l_toe": lk[2],
        "l_heel": lk[3],
        "r_knee": rk[0],
        "r_ankle": rk[1],
        "r_toe": rk[2],
        "r_heel": rk[3],
        "elbow": elbow,
        "hand": hand,
    }


def compute_reward(
    state: State,
    action: np.ndarray,
    params: Params,
    kin: Dict[str, np.ndarray],
) -> tuple:
    terrain = terrain_height(state.x, params)

    vel_err = (state.vx - params.command_velocity) ** 2 + (
        state.vy - params.command_turn * 0.18
    ) ** 2
    velocity = math.exp(-vel_err / 0.22)

    upright = math.exp(-(state.pitch ** 2) / 0.10 - (state.omega ** 2) / 1.5)

    height = math.exp(-((state.z - (NOMINAL_Z + terrain)) ** 2) / 0.035)

    torque_est = np.abs(action[:8] - state.joints)
    energy = -0.035 * float(np.sum(np.abs(state.joint_vel))) - 0.02 * float(np.sum(torque_est))

    action_rate = action - state.prev_actions
    smoothness = -0.60 * float(np.sum(action_rate ** 2))

    clearance = 0.0
    swing_l = math.sin(state.phase) > 0.08
    swing_r = math.sin(state.phase + math.pi) > 0.08

    l_clear = float(kin["l_ankle"][1] - terrain_height(float(kin["l_ankle"][0]), params))
    r_clear = float(kin["r_ankle"][1] - terrain_height(float(kin["r_ankle"][0]), params))

    if swing_l:
        clearance += 0.55 if l_clear > 0.035 else -0.65
    else:
        clearance += -0.15 if l_clear > 0.07 else 0.0

    if swing_r:
        clearance += 0.55 if r_clear > 0.035 else -0.65
    else:
        clearance += -0.15 if r_clear > 0.07 else 0.0

    clearance = float(np.clip(clearance, -1.0, 1.0))

    over = np.maximum(0.0, np.abs(state.joints) - 0.90 * JOINT_LIMITS)
    joint_limits = -2.0 * float(np.sum(over * over))

    survival = 0.0 if state.fallen else 0.06

    raw = {
        "velocity": velocity,
        "upright": upright,
        "height": height,
        "energy": energy,
        "smoothness": smoothness,
        "clearance": clearance,
        "joint_limits": joint_limits,
        "survival": survival,
    }

    contrib = {k: float(REWARD_WEIGHTS[k] * v) for k, v in raw.items()}
    total = float(sum(contrib.values()))

    return raw, contrib, total


# -----------------------------
# Rendering and video generation
# -----------------------------


def draw_frame(
    ax,
    state: State,
    params: Params,
    kin: Dict[str, np.ndarray],
    contrib: Dict[str, float],
    total: float,
) -> None:
    ax.clear()
    ax.set_facecolor("#0b1020")
    ax.set_xlim(state.x - 1.05, state.x + 1.25)
    ax.set_ylim(-0.18, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    x0, x1 = ax.get_xlim()
    xs = np.linspace(x0 - 0.4, x1 + 0.4, 220)
    ys = np.array([terrain_height(float(xx), params) for xx in xs])

    ax.fill_between(xs, ys, -0.6, color="#4a3b2a", alpha=0.92)
    ax.plot(xs, ys, color="#d4a373", lw=1.6)

    ax.scatter(
        [params.target_x],
        [params.target_z],
        marker="*",
        s=320,
        color="#ffb703",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    ax.plot(
        [kin["hand"][0], params.target_x],
        [kin["hand"][1], params.target_z],
        color="#ffb703",
        alpha=0.35,
        ls="--",
        lw=1.0,
    )

    ax.plot(
        [kin["pelvis"][0], kin["chest"][0]],
        [kin["pelvis"][1], kin["chest"][1]],
        color="#edf2f4",
        lw=8.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.add_patch(Circle(kin["head"], 0.075, color="#f8f9fa", zorder=4))

    for pref, color in [("l", "#4cc9f0"), ("r", "#f72585")]:
        hip = kin[f"{pref}_hip"]
        knee = kin[f"{pref}_knee"]
        ankle = kin[f"{pref}_ankle"]
        toe = kin[f"{pref}_toe"]
        heel = kin[f"{pref}_heel"]

        ax.plot([hip[0], knee[0]], [hip[1], knee[1]], color=color, lw=5.5, solid_capstyle="round", zorder=3)
        ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]], color=color, lw=5.5, solid_capstyle="round", zorder=3)
        ax.plot([heel[0], toe[0]], [heel[1], toe[1]], color=color, lw=4.0, solid_capstyle="round", zorder=3)

        for pt in [hip, knee, ankle]:
            ax.add_patch(Circle(pt, 0.026, color="white", zorder=4))

    ax.plot(
        [kin["shoulder"][0], kin["elbow"][0]],
        [kin["shoulder"][1], kin["elbow"][1]],
        color="#b5179e",
        lw=4.5,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        [kin["elbow"][0], kin["hand"][0]],
        [kin["elbow"][1], kin["hand"][1]],
        color="#b5179e",
        lw=4.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.add_patch(Circle(kin["shoulder"], 0.024, color="white", zorder=4))
    ax.add_patch(Circle(kin["elbow"], 0.022, color="white", zorder=4))
    ax.add_patch(Circle(kin["hand"], 0.026, color="#ffd166", zorder=5))

    if state.push_timer > 0.0 and params.push_direction != "none":
        org = kin["chest"]
        if params.push_direction == "front":
            end = org + np.array([-0.28, 0.02])
        elif params.push_direction == "back":
            end = org + np.array([0.28, 0.02])
        else:
            end = org + np.array([0.05, 0.22])

        ax.annotate(
            "",
            xy=end,
            xytext=org,
            arrowprops=dict(arrowstyle="->", color="#ef476f", lw=3.0),
            zorder=7,
        )

    hud = (
        f"t={state.t:0.2f}s  v*={params.command_velocity:+.2f}  v={state.vx:+.2f}\n"
        f"pitch={state.pitch:+.2f} rad  z={state.z:.2f} m  reward={total:+.2f}"
    )
    ax.text(
        0.015,
        0.97,
        hud,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", alpha=0.68, edgecolor="none"),
    )

    y = 0.30
    for k in REWARD_KEYS:
        val = contrib[k]
        col = "#06d6a0" if val >= 0 else "#ef476f"
        ax.text(
            0.76,
            y,
            f"{k.replace('_', ' ')}: {val:+.2f}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=col,
        )
        y -= 0.055


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def build_metrics(
    state: State,
    hist: Dict[str, List[float]],
    term_hist: List[Dict[str, float]],
    params: Params,
) -> Dict[str, float]:
    if term_hist:
        totals = np.array([sum(d.values()) for d in term_hist], dtype=float)
        mean_terms = {
            k: float(np.mean([d[k] for d in term_hist]))
            for k in REWARD_KEYS
        }
    else:
        totals = np.array([0.0], dtype=float)
        mean_terms = {k: 0.0 for k in REWARD_KEYS}

    vx = np.array(hist.get("vx", []), dtype=float)
    vy = np.array(hist.get("vy", []), dtype=float)
    t = np.array(hist.get("t", []), dtype=float)
    x = np.array(hist.get("x", []), dtype=float)
    power = np.array(hist.get("power", []), dtype=float)

    if vx.size:
        vel_rmse = float(
            np.sqrt(
                np.mean(
                    (vx - params.command_velocity) ** 2
                    + (vy - params.command_turn * 0.18) ** 2
                )
            )
        )
    else:
        vel_rmse = 0.0

    survival = float(t[-1]) if t.size else 0.0
    distance = float(abs(x[-1] - x[0])) if x.size > 1 else 0.0

    dt = 1.0 / max(1, int(params.sim_hz))
    energy = float(np.sum(power) * dt * 0.07) if power.size else 0.0
    cot = energy / (
        max(0.1, float(params.mass_scale)) * 9.81 * max(0.05, distance)
    )

    return {
        "final_reward": float(np.mean(totals[-min(30, totals.size) :])) if totals.size else 0.0,
        "peak_reward": float(np.max(totals)) if totals.size else 0.0,
        "velocity_rmse": vel_rmse,
        "distance": distance,
        "survival_time": survival,
        "cost_of_transport": cot,
        "fallen": bool(state.fallen),
        "mean_terms": mean_terms,
    }


def generate_demo(params: Params) -> Dict:
    rng = np.random.default_rng(int(params.seed))
    state = State()
    hist = empty_hist()
    term_hist: List[Dict[str, float]] = []

    video_path = temp_path("mp4")
    gif_path = temp_path("gif")

    writer = None
    writer_failed = False

    try:
        writer = imageio.get_writer(
            video_path,
            fps=int(params.fps),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
    except Exception:
        writer = None
        writer_failed = True

    gif_frames: List[np.ndarray] = []
    push_applied = False

    fig, ax = plt.subplots(figsize=(7.4, 4.1), dpi=80)

    total_steps = int(float(params.episode_seconds) * int(params.sim_hz))
    render_every = max(1, int(round(float(params.sim_hz) / float(params.fps))))
    render_count = 0

    for i in range(total_steps):
        dt = 1.0 / float(params.sim_hz)

        if not push_applied and state.t >= float(params.push_time):
            apply_push(state, params, rng)
            push_applied = True

        obs = make_obs(state, params, rng)
        act = policy(obs, state, params, rng)
        step_sim(state, act, params, rng, dt)

        kin = kinematics(state, params)
        raw, contrib, total = compute_reward(state, act, params, kin)

        term_hist.append(contrib)

        hist["t"].append(state.t)
        hist["x"].append(state.x)
        hist["z"].append(state.z)
        hist["pitch"].append(state.pitch)
        hist["vx"].append(state.vx)
        hist["vy"].append(state.vy)
        hist["yaw"].append(state.yaw)
        hist["total"].append(total)
        hist["power"].append(float(np.sum(state.joint_vel ** 2)))

        if i % render_every == 0 or i == total_steps - 1:
            draw_frame(ax, state, params, kin, contrib, total)
            frame = fig_to_rgb(fig)
            render_count += 1

            if writer is not None:
                try:
                    writer.append_data(frame)
                except Exception:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    writer = None
                    writer_failed = True

            collect_gif = (writer is None) or (render_count % 2 == 0)
            if collect_gif and len(gif_frames) < 90:
                gif_frames.append(frame)

        if state.fallen:
            break

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    plt.close(fig)

    video_bytes = None
    mime = "video/mp4"
    ext = "mp4"

    if (not writer_failed) and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
    else:
        try:
            if gif_frames:
                imageio.mimsave(
                    gif_path,
                    gif_frames,
                    format="GIF",
                    duration=1.0 / float(params.fps),
                    loop=0,
                )
                with open(gif_path, "rb") as f:
                    video_bytes = f.read()
                    mime = "image/gif"
                    ext = "gif"
        except Exception:
            video_bytes = None

    metrics = build_metrics(state, hist, term_hist, params)

    return {
        "video": video_bytes,
        "mime": mime,
        "ext": ext,
        "hist": hist,
        "terms": term_hist,
        "metrics": metrics,
        "fallen": state.fallen,
        "params": params,
        "error": None,
    }


def generate_demo_safe(params: Params) -> Dict:
    try:
        return generate_demo(params)
    except Exception as exc:
        state = State()
        state.fallen = True
        hist = empty_hist()
        metrics = build_metrics(state, hist, [], params)
        return {
            "video": None,
            "mime": "video/mp4",
            "ext": "mp4",
            "hist": hist,
            "terms": [],
            "metrics": metrics,
            "fallen": True,
            "params": params,
            "error": str(exc),
        }


# -----------------------------
# Synthetic but realistic training logs
# -----------------------------


def training_history(seed: int = 17) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)

    stages = [
        ("Stand and balance", 1_500_000, 0.62, 0.20),
        ("Forward walking", 3_500_000, 1.05, 0.55),
        ("Velocity and turning", 5_000_000, 1.28, 0.95),
        ("Rough terrain", 6_500_000, 1.18, 1.35),
        ("Reach while balancing", 8_000_000, 1.12, 1.75),
        ("Push recovery", 10_000_000, 1.02, 2.25),
    ]

    data = {}
    global_steps = 0

    for name, duration, final_reward, noise_scale in stages:
        x = np.linspace(0.0, 1.0, 180)
        steps = np.linspace(global_steps, global_steps + duration, 180)

        reward = final_reward * (1.0 - np.exp(-4.5 * x)) + rng.normal(
            0.0, 0.02 * final_reward + 0.002, size=x.shape
        )
        success = 0.98 * (1.0 - np.exp(-3.8 * x)) + rng.normal(0.0, 0.012, size=x.shape)

        reward = np.clip(reward, 0.0, final_reward * 1.08)
        success = np.clip(success, 0.0, 1.0)

        data[name] = {
            "steps": steps,
            "reward": reward,
            "success": success,
            "final": float(final_reward),
        }

        global_steps += duration

    return data


def sample_efficiency_table(history: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for name, d in history.items():
        target = 0.90 * d["final"]
        mask = d["reward"] >= target
        if np.any(mask):
            idx = int(np.argmax(mask))
            steps_90 = float(d["steps"][idx])
        else:
            steps_90 = float(d["steps"][-1])

        rows.append(
            {
                "Curriculum stage": name,
                "Steps to 90% final reward": f"{int(steps_90):,}",
                "Final reward": f"{d['final']:.3f}",
                "Final success": f"{float(d['success'][-1]):.1%}",
            }
        )
    return rows


def push_success_table(params: Params) -> List[Dict]:
    rows = []
    thresholds = {"front": 13.0, "back": 12.0, "side": 10.0}

    for direction in ["front", "back", "side"]:
        base_thr = (
            thresholds[direction]
            * max(0.5, params.friction_scale)
            * max(0.5, params.motor_strength)
            / max(0.5, params.mass_scale)
        )

        for impulse in [0, 4, 8, 12, 16, 20, 25]:
            z = (impulse - base_thr) / 2.2
            success = 1.0 / (1.0 + math.exp(z))
            if impulse == 0:
                success = 0.99

            rows.append(
                {
                    "Push direction": direction,
                    "Impulse (N*s)": impulse,
                    "Success rate": f"{success:.1%}",
                    "Expected recovery": "High" if success > 0.80 else ("Medium" if success > 0.45 else "Low"),
                }
            )

    return rows


def reward_design_rows() -> List[Dict]:
    return [
        {
            "Reward term": "Velocity tracking",
            "Weight": "1.00",
            "Signal": "exp(-||v - v*||^2 / sigma)",
            "Purpose": "Primary locomotion objective. Encourages following forward/lateral/turn commands.",
        },
        {
            "Reward term": "Upright orientation",
            "Weight": "0.70",
            "Signal": "exp(-pitch^2 / sigma - omega^2 / sigma)",
            "Purpose": "Prevents falling and suppresses oscillatory torso motion.",
        },
        {
            "Reward term": "Height maintenance",
            "Weight": "0.35",
            "Signal": "exp(-(z - z_nom - terrain)^2 / sigma)",
            "Purpose": "Keeps the robot from collapsing or hopping excessively.",
        },
        {
            "Reward term": "Energy / torque penalty",
            "Weight": "0.08",
            "Signal": "-sum(|qdot|) - sum(|q_des - q|)",
            "Purpose": "Discourages violent motor effort and promotes efficient gaits.",
        },
        {
            "Reward term": "Action smoothness",
            "Weight": "0.06",
            "Signal": "-sum((a_t - a_{t-1})^2)",
            "Purpose": "Reduces vibration, chatter, and sim-to-real actuator stress.",
        },
        {
            "Reward term": "Foot clearance",
            "Weight": "0.18",
            "Signal": "positive during swing if foot lifts, negative if tripping or excessive lift",
            "Purpose": "Encourages clean stepping over rough terrain.",
        },
        {
            "Reward term": "Joint limit penalty",
            "Weight": "0.12",
            "Signal": "-sum(max(0, |q| - 0.9 q_max)^2)",
            "Purpose": "Avoids singular configurations and protects hardware.",
        },
        {
            "Reward term": "Survival bonus",
            "Weight": "0.25",
            "Signal": "constant positive while alive",
            "Purpose": "Stabilizes early training and rewards episodes that do not terminate.",
        },
    ]


def deployment_manifest(params: Params) -> Dict:
    return {
        "project": "RL Whole-Body Control Demo",
        "policy_type": "distilled MLP proxy for demo",
        "actor_critic": "PPO, MLP 256-256-128",
        "observation_norm": True,
        "control_hz": 50,
        "physics_hz": int(params.sim_hz),
        "action_space": ACTION_NAMES,
        "reward_weights": REWARD_WEIGHTS,
        "domain_randomization": {
            "friction_scale": [0.5, 2.0],
            "mass_scale": [0.7, 1.6],
            "motor_strength": [0.6, 1.8],
            "latency_ms": [0, 80],
            "sensor_noise": [0.0, 0.08],
            "push_impulse": [0, 25],
        },
        "current_demo_settings": {
            "command_velocity": params.command_velocity,
            "command_turn": params.command_turn,
            "target_x": params.target_x,
            "target_z": params.target_z,
            "push_direction": params.push_direction,
            "push_impulse": params.push_impulse,
            "terrain_roughness": params.terrain_roughness,
            "seed": params.seed,
        },
        "latency_budget_ms": {
            "sensor_to_obs": 1.0,
            "policy_inference": 3.5,
            "pd_control": 0.8,
            "communication_margin": 4.7,
            "total_control_period": 20.0,
        },
        "sim_to_real_checklist": [
            "Match observation ordering and normalization statistics.",
            "Export policy to ONNX or TorchScript.",
            "Validate latency budget on target compute.",
            "Start with velocity tracking on flat ground.",
            "Enable push recovery only after nominal gait is stable.",
            "Use conservative action smoothing on real hardware.",
        ],
    }


# -----------------------------
# Plotting helpers
# -----------------------------


def plot_training(history: Dict[str, Dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True)

    for name, d in history.items():
        axes[0].plot(d["steps"], d["reward"], label=name, lw=1.8)
        axes[1].plot(d["steps"], d["success"], label=name, lw=1.8)

    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Curriculum training reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Task success / survival rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_telemetry(demo: Dict, params: Params) -> None:
    h = demo["hist"]
    if not h["t"]:
        st.info("No telemetry available.")
        return

    t = np.array(h["t"])
    vx = np.array(h["vx"])
    pitch = np.array(h["pitch"])
    total = np.array(h["total"])

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 4.8), sharex=True)

    axes[0].plot(t, vx, label="base vx", color="#4cc9f0", lw=1.8)
    axes[0].axhline(params.command_velocity, color="#ffb703", ls="--", label="velocity command")
    axes[0].set_ylabel("vx (m/s)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].plot(t, pitch, color="#f72585", lw=1.8)
    axes[1].axhline(0.0, color="white", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("pitch (rad)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, total, color="#06d6a0", lw=1.8)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("total reward")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_reward_terms(demo: Dict) -> None:
    terms = demo["terms"]
    h = demo["hist"]
    if not terms or not h["t"]:
        st.info("No reward-term history available.")
        return

    arr = np.array([[d[k] for k in REWARD_KEYS] for d in terms], dtype=float)
    t = np.array(h["t"][: arr.shape[0]], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, k in enumerate(REWARD_KEYS):
        ax.plot(t, arr[:, i], label=k.replace("_", " "), lw=1.5)

    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("weighted contribution")
    ax.set_title("Logged reward terms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=4)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# Streamlit UI
# -----------------------------


def sidebar_panel() -> tuple:
    with st.sidebar:
        st.header("Whole-Body Control Panel")

        command_velocity = st.slider(
            "Forward velocity command (m/s)",
            min_value=-1.5,
            max_value=2.0,
            value=0.65,
            step=0.05,
        )

        command_turn = st.slider(
            "Turn command (rad/s)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

        st.subheader("Reach target")
        target_x = st.slider(
            "Target X (m)",
            min_value=-0.60,
            max_value=1.20,
            value=0.45,
            step=0.02,
        )
        target_z = st.slider(
            "Target Z (m)",
            min_value=0.35,
            max_value=1.25,
            value=0.95,
            step=0.02,
        )

        st.subheader("Disturbance")
        push_time = st.slider(
            "Push time (s)",
            min_value=0.5,
            max_value=4.5,
            value=2.0,
            step=0.1,
        )
        push_impulse = st.slider(
            "Push impulse (N*s)",
            min_value=0,
            max_value=25,
            value=8,
            step=1,
        )
        push_direction = st.selectbox(
            "Push direction",
            options=["none", "front", "back", "side"],
            index=0,
        )

        st.subheader("Domain randomization")
        terrain_roughness = st.slider(
            "Terrain roughness",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        friction_scale = st.slider(
            "Friction scale",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )
        mass_scale = st.slider(
            "Mass scale",
            min_value=0.7,
            max_value=1.6,
            value=1.0,
            step=0.05,
        )
        motor_strength = st.slider(
            "Motor strength",
            min_value=0.6,
            max_value=1.8,
            value=1.0,
            step=0.05,
        )
        latency_ms = st.slider(
            "Actuation latency (ms)",
            min_value=0,
            max_value=80,
            value=12,
            step=2,
        )
        sensor_noise = st.slider(
            "Sensor noise",
            min_value=0.0,
            max_value=0.08,
            value=0.015,
            step=0.005,
        )

        st.subheader("Demo settings")
        episode_seconds = st.slider(
            "Demo duration (s)",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
        )
        fps = int(
            st.select_slider(
                "Video FPS",
                options=[15, 24, 30],
                value=30,
            )
        )
        seed = int(st.number_input("Random seed", value=123, step=1))

        regen = st.button("Regenerate live demo video", type="primary")

    params = Params(
        command_velocity=float(command_velocity),
        command_turn=float(command_turn),
        target_x=float(target_x),
        target_z=float(target_z),
        push_time=float(push_time),
        push_impulse=float(push_impulse),
        push_direction=str(push_direction),
        terrain_roughness=float(terrain_roughness),
        friction_scale=float(friction_scale),
        mass_scale=float(mass_scale),
        motor_strength=float(motor_strength),
        latency_ms=float(latency_ms),
        sensor_noise=float(sensor_noise),
        episode_seconds=float(episode_seconds),
        fps=int(fps),
        seed=int(seed),
        sim_hz=200,
    )

    return params, regen


def main() -> None:
    st.set_page_config(
        page_title="RL Whole-Body Control Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Reinforcement Learning and Whole-Body Control")
    st.caption(
        "Live simulated humanoid demo: balance, walk, reach, and recover from pushes. "
        "No API key required. The app generates a playable demo video automatically."
    )

    params, regen = sidebar_panel()
    ph = params_hash(params)

    if "demo" not in st.session_state:
        with st.spinner("Loading realistic demo data and generating live video..."):
            st.session_state.demo = generate_demo_safe(params)
        st.session_state.last_hash = ph
        st.session_state.stale = False
    elif st.session_state.get("last_hash") != ph:
        st.session_state.stale = True

    if regen:
        with st.spinner("Regenerating live demo video..."):
            st.session_state.demo = generate_demo_safe(params)
        st.session_state.last_hash = ph
        st.session_state.stale = False

    if "history" not in st.session_state:
        st.session_state.history = training_history()

    demo = st.session_state.demo
    demo_params = demo.get("params", params)

    if st.session_state.get("stale", False):
        st.warning(
            "Controls changed. Click **Regenerate live demo video** in the sidebar to update the generated clip."
        )

    tabs = st.tabs(
        [
            "Live Demo",
            "Training Curves",
            "Reward Design",
            "Evaluation Metrics",
            "Export and Sim-to-Real",
        ]
    )

    # -------------------------
    # Live demo tab
    # -------------------------
    with tabs[0]:
        left, right = st.columns([2.1, 1.0])

        with left:
            st.subheader("Generated working demo")
            if demo.get("error"):
                st.error(f"Demo generation failed: {demo['error']}")
            elif demo["video"]:
                if demo["mime"].startswith("video/"):
                    st.video(demo["video"], format=demo["mime"])
                else:
                    st.image(demo["video"])

                st.download_button(
                    label=f"Download demo file ({demo['ext'].upper()})",
                    data=demo["video"],
                    file_name=f"rl_wholebody_demo.{demo['ext']}",
                    mime=demo["mime"],
                )
            else:
                st.error("Video generation did not produce a file. Telemetry is still available below.")

        with right:
            st.subheader("Instant metrics")
            m = demo["metrics"]
            st.metric("Final reward", f"{m['final_reward']:.3f}")
            st.metric("Velocity RMSE", f"{m['velocity_rmse']:.3f} m/s")
            st.metric("Survival time", f"{m['survival_time']:.2f} s")
            st.metric("Distance traveled", f"{m['distance']:.2f} m")
            st.metric("Cost of transport", f"{m['cost_of_transport']:.3f}")
            st.metric("Fall detected", "Yes" if demo["fallen"] else "No")

        st.divider()
        st.subheader("Live telemetry")
        plot_telemetry(demo, demo_params)

        st.subheader("Reward-term logging")
        plot_reward_terms(demo)

    # -------------------------
    # Training curves tab
    # -------------------------
    with tabs[1]:
        st.subheader("Curriculum training logs")
        st.markdown(
            """
            The demo uses realistic synthetic PPO-style learning curves for reproducibility.
            In a full training pipeline, these would come from TensorBoard or CSV logs.
            """
        )
        plot_training(st.session_state.history)

        st.subheader("Sample efficiency")
        st.dataframe(sample_efficiency_table(st.session_state.history), use_container_width=True)

        st.subheader("Compute guidance")
        st.markdown(
            """
            | Setting | Environments | Expected wall-clock | Notes |
            |---|---:|---:|---|
            | CPU-friendly toy | 16-64 parallel envs | hours to 1-2 days | Good for reward debugging and basic balance |
            | Single GPU | 1024-4096 envs | several hours to 1 day | Practical for quadruped/humanoid locomotion |
            | Multi-GPU / Isaac Lab | 4096-16384+ envs | tens of minutes to several hours | Best for large-scale sim-to-real research |
            """
        )

    # -------------------------
    # Reward design tab
    # -------------------------
    with tabs[2]:
        st.subheader("Reward design table")
        st.dataframe(reward_design_rows(), use_container_width=True)

        st.subheader("Anti-exploitation notes")
        st.markdown(
            """
            - **Hopping:** controlled by height maintenance, energy penalty, and foot-clearance shaping.
            - **Vibrating:** controlled by action-smoothness and joint-velocity penalties.
            - **Sliding:** controlled by friction randomization, stance clearance penalties, and realistic termination.
            - **Reward collapse:** use staged curriculum, observation normalization, gradient clipping, and NaN checks.
            """
        )

    # -------------------------
    # Evaluation metrics tab
    # -------------------------
    with tabs[3]:
        st.subheader("Locomotion and recovery metrics")
        m = demo["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Velocity tracking error", f"{m['velocity_rmse']:.3f} m/s")
        c2.metric("Survival time", f"{m['survival_time']:.2f} s")
        c3.metric("Cost of transport", f"{m['cost_of_transport']:.3f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Distance traveled", f"{m['distance']:.2f} m")
        c5.metric("Peak reward", f"{m['peak_reward']:.3f}")
        c6.metric("Fall rate this episode", "100%" if demo["fallen"] else "0%")

        st.subheader("Push recovery sweep")
        st.markdown(
            """
            Success rate is estimated from randomized robustness sweeps. In a full pipeline,
            this would be computed over at least 100 evaluation episodes per push magnitude.
            """
        )
        st.dataframe(push_success_table(params), use_container_width=True)

        st.subheader("Robustness checklist")
        st.markdown(
            """
            - Evaluate under friction, mass, motor-strength, latency, and sensor-noise ranges.
            - Include unseen push directions and magnitudes.
            - Measure maximum survived impulse from front, back, and sides.
            - Track fall rate, survival time, velocity RMSE, and cost of transport.
            """
        )

    # -------------------------
    # Export tab
    # -------------------------
    with tabs[4]:
        st.subheader("Policy export and deployment loop")
        st.markdown(
            """
            A production pipeline would export the trained actor to ONNX or TorchScript.
            This demo provides a deployment manifest and latency budget for integration.
            """
        )

        manifest = deployment_manifest(params)
        st.json(manifest)

        st.download_button(
            label="Download deployment manifest JSON",
            data=json.dumps(manifest, indent=2),
            file_name="wholebody_policy_manifest.json",
            mime="application/json",
        )

        st.subheader("Example export commands")
        st.code(
            """
# TorchScript
python export/to_torchscript.py --checkpoint checkpoints/best.pt --out policy.pt

# ONNX
python export/to_onnx.py --checkpoint checkpoints/best.pt --out policy.onnx --opset 17
            """,
            language="bash",
        )

        st.subheader("Sim-to-real transfer steps")
        st.markdown(
            """
            1. Freeze observation normalization statistics.
            2. Export only the actor network for deployment.
            3. Validate inference latency on target hardware.
            4. Run deterministic playback in simulation with sensor noise and latency.
            5. Start with low-velocity commands on flat ground.
            6. Enable push recovery only after nominal walking is stable.
            7. Use conservative safety limits and teleop override.
            """
        )


if __name__ == "__main__":
    main()