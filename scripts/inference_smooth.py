import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import logging
import multiprocessing as mp
import collections
import yaml
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# ---------- 子进程：推理循环 ----------
def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    config,
    checkpoint_dir,
):

    # 1. 只在该进程里加载一次模型 / CUDA
    data_config = DATA_CONFIG_MAP[config.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    policy: BasePolicy = Gr00tPolicy(
        model_path=config.checkpoint_dir,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    while True:
        item = in_q.get()
        if item is None:            # 收到结束标识
            del policy
            break
        idx, obs = item       # idx 用来对应主进程里的顺序
        start_time = time.time()
        result = policy.get_action(obs)
        infer_time = time.time() - start_time
        print(f"Step {idx}: infer time = {infer_time:.4f} seconds")
        action = np.concatenate(
            [result['action.arm_joints'],  # (16,6)
             result['action.gripper'][:, None]],  # (16,) → (16,1)
            axis=1  # 沿列方向拼
        )
        out_q.put((idx, action))

def linear_transition(old_actions, new_actions):
    """
    线性插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray:shape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        interp_action = (1 - t) * old_actions[i_interp] + t * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def cubic_transition(old_actions, new_actions):
    """
    三次插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarrayshape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        # 三次Hermite插值（ease in/out）：h(t) = 3t^2 - 2t^3
        h = 3 * t**2 - 2 * t**3
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def quintic_transition(old_actions, new_actions):
    """
    五次多项式插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        # 五次多项式插值：h(t) = 10t^3 - 15t^4 + 6t^5
        h = 10 * t**3 - 15 * t**4 + 6 * t**5
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def ema_transition(old_actions, new_actions, alpha=0.7):
    """
    指数加权平滑(EMA)，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    alpha: 新动作权重,0~1
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        interp_action = alpha * new_actions[i_interp] + (1 - alpha) * old_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def main():
    parser = argparse.ArgumentParser(description="Inference script for AgileX follower robot")
    parser.add_argument("--port", type=str, required=True, help="port name")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="path to checkpoint directory")
    parser.add_argument("--fps", type=int, required=False, default=30, help="frames per second")
    parser.add_argument("--task", type=str, required=False, help="task prompt", default="pick up the circular chip and place it on the yellow pot")
    parser.add_argument("--id", type=str, required=False, help="robot id", default="left")
    parser.add_argument("--cameras", type=str, required=False, help="camera config yaml", default=None)
    parser.add_argument("--max_relative_target", type=int, required=False, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    parser.add_argument("--action_steps", type=int, required=False, default=20, help="number of action steps to execute before next inference")
    parser.add_argument("--smooth_type", type=str, default="cubic", choices=["linear", "cubic", "quintic", "ema"], help="动作平滑策略: linear/cubic/quintic/ema")
    parser.add_argument("--ema_alpha", type=float, default=0.7, help="EMA平滑时新动作权重alpha,0~1")
    parser.add_argument("--align_mode", type=str, default="step", choices=["step", "euclidean"], help="新动作对齐方式: step(步数) 或 euclidean(欧氏距离)")
    parser.add_argument("--data_config", type=str, default="aloha_single_arm", help="for gr00t")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment", help="for gr00t")
    parser.add_argument("--denoising_steps", type=int, default=10, help="for gr00t")

    args = parser.parse_args()

    # 解析摄像头配置
    if args.cameras is not None:
        raw = yaml.safe_load(args.cameras)
        cameras = {name: OpenCVCameraConfig(index_or_path=cfg['index_or_path'], width=640, height=480, fps=30) for name, cfg in raw.items()}
    else:
        cameras = {}


    robot_config = AlohaAgileXFollowerConfig(
        port=args.port,
        id=args.id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
    )
    # 选择配置和 checkpoint
    robot = AlohaAgileXFollower(robot_config)

    # Load pretrained policy
    # policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # ==== 1. 启动推理子进程 ====
    ctx = mp.get_context("spawn")        # "spawn" 更安全，尤其 CUDA
    in_q: mp.Queue = ctx.Queue(maxsize=4)   # 根据实时性调节 maxsize
    out_q: mp.Queue = ctx.Queue(maxsize=4)
    checkpoint_dir = args.checkpoint_dir #"/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")

    proc = ctx.Process(
        target=inference_worker,
        args=(in_q, out_q, args, checkpoint_dir)
    )
    proc.daemon = True
    proc.start()

    robot.connect()
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 6000

    # rows = []
    step = 1
    step_time = step/(args.fps)
    # prompt = args.task
    # tokenizer = PaligemmaTokenizer()
    # tokenized, mask = tokenizer.tokenize(prompt)

    action_queue = collections.deque()  # 存储当前动作序列
    waiting_for_infer = False
    action_step_counter = 0  # 记录已执行的动作步数
    first = True

    while i < kMaxTimeStamps:
        t0 = time.perf_counter()

        # 1. 只有在执行了action_steps步后才采集观测并推理
        if not waiting_for_infer and (action_step_counter >= args.action_steps or first):
            first = False
            obs = robot.get_observation()
            # first add the images
            obs_dict = {'video.middle_view': obs['images']['camera0'], 'video.right_view': obs['images']['camera1'],
                        'video.left_view': obs['images']['camera2'],
                        "state.arm_joints": obs['state'][:6].astype(np.float64),
                        "state.gripper": obs['state'][6:7].astype(np.float64),
                        "annotation.human.task_description": "pick up the circular chip and place it on the yellow pot"}

            # # show images
            # if self.show_images:
            #     view_img(obs_dict)

            # then add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
            for k in obs_dict:
                if isinstance(obs_dict[k], np.ndarray):
                    obs_dict[k] = obs_dict[k][np.newaxis, ...]
                else:
                    obs_dict[k] = [obs_dict[k]]

            try:
                in_q.put_nowait((sent_idx, obs_dict))
                sent_idx += 1
                waiting_for_infer = True
                action_step_counter = 0
            except mp.queues.Full:
                logging.debug("inference queue full, dropping frame")

        # 2. 如果有新推理结果，立即清空并更新 action_queue
        try:
            idx, action_vals = out_q.get_nowait()
            recv_idx = idx
            logging.debug(f"got result #{recv_idx}")

            # 1. 记录未执行的旧动作
            old_actions = list(action_queue)
            action_queue.clear()

            # 2. 新推理动作起点
            if args.align_mode == "step":
                start_idx = action_step_counter
            elif args.align_mode == "euclidean" and len(old_actions) > 0 and len(action_vals) > 0:
                # 取旧队列第一个动作，与新动作序列做欧氏距离最小匹配
                old_action = old_actions[0]
                dists = np.linalg.norm(action_vals - old_action, axis=1)
                start_idx = int(np.argmin(dists))
            else:
                start_idx = 0
            new_actions = action_vals[start_idx:]

            # 3. 平滑衔接（可通过参数切换）
            if args.smooth_type == "linear":
                smooth_actions = linear_transition(old_actions, new_actions)
            elif args.smooth_type == "cubic":
                smooth_actions = cubic_transition(old_actions, new_actions)
            elif args.smooth_type == "quintic":
                smooth_actions = quintic_transition(old_actions, new_actions)
            elif args.smooth_type == "ema":
                smooth_actions = ema_transition(old_actions, new_actions, alpha=args.ema_alpha)
            else:
                raise ValueError(f"Unknown smooth_type: {args.smooth_type}")
            for a in smooth_actions:
                action_queue.append(a)

            waiting_for_infer = False
        except mp.queues.Empty:
            pass

        # 3. 如果 action_queue 有动作，发给 robot
        if action_queue:
            action_to_send = action_queue.popleft()
            robot.send_action_np(action_to_send[:7])
            action_step_counter += 1
            # print(f'publish an action:{time.perf_counter()},action counter:{action_step_counter}')

        # 2.5 统计
        i += 1
        dt_s = time.perf_counter() - t0
        # print(f"loop {i} dt={dt_s:.3f} s")
        time.sleep(max(step_time - dt_s,0))

    # ==== 3. 结束 ====
    in_q.put(None)      # 通知子进程退出
    proc.join()
    robot.disconnect()
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 6000

if __name__ == "__main__":
    main()