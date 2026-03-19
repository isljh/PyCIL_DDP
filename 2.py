import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def get_norms(path):
    if not os.path.exists(path):
        return None
    state_dict = torch.load(path, map_location='cpu')
    # 自动识别键名
    key = 'fc.weight' if 'fc.weight' in state_dict else 'module.fc.weight'
    if key not in state_dict:
        return None
    weights = state_dict[key]
    return torch.norm(weights, p=2, dim=1).numpy()


def analyze_comparison(path_old, path_new):
    norms_old = get_norms(path_old)
    norms_new = get_norms(path_new)

    plt.figure(figsize=(14, 7))

    # 绘制新任务的权重（它包含了所有已学过的类）
    if norms_new is not None:
        num_classes = len(norms_new)
        plt.bar(range(num_classes), norms_new, color='salmon', alpha=0.6, label='Current Task (New)')

    # 在同一张图上重叠绘制旧任务的权重（观察变化）
    if norms_old is not None:
        plt.bar(range(len(norms_old)), norms_old, color='royalblue', alpha=0.4, label='Previous Task (Old)')

    # 标记任务分界线（假设每 10 个类一个任务）
    for i in range(10, len(norms_new) if norms_new is not None else 0, 10):
        plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)

    plt.title("Weight Norm Comparison: Old vs New Tasks", fontsize=15)
    plt.xlabel("Class Index", fontsize=12)
    plt.ylabel("L2 Norm", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    save_path = "weight_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"对比图已生成: {save_path}")


# --- 修改这里的路径 ---
path_0 = "logs/tagfex/imagenet100_lejepa/0/10/20260202_171458/checkpoints/reproduce_1993_task_0.pth"
# 等你跑完 Task 1 后，把下面的路径改正确
path_1 = "logs/tagfex/imagenet100_lejepa/0/10/20260202_171458/checkpoints/reproduce_1993_task_1.pth"

analyze_comparison(path_0, path_1)