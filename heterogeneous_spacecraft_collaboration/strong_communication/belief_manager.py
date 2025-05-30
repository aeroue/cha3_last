# strong_communication/belief_manager.py
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

# --- 临时路径处理 ---
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(os.path.dirname(_current_file_directory))
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)
# --- 结束 ---

class BeliefManager:
    """
    管理和更新多个航天器对多个任务类型的信念。
    信念使用狄利克雷分布的参数（伪计数）表示。
    """
    def __init__(self,
                 agent_ids: List[str],
                 task_ids: List[str],
                 num_task_types: int,
                 initial_belief_alpha: float = 1.0): # 初始伪计数 (例如，均匀先验)
        """
        初始化信念管理器。

        参数:
            agent_ids (List[str]): 所有航天器的ID列表。
            task_ids (List[str]): 所有任务的ID列表。
            num_task_types (int): 可能的任务类型数量。
            initial_belief_alpha (float): 用于狄利克雷分布的初始伪计数参数。
                                         默认为1.0，表示每个类型都有一个初始计数，
                                         对应于均匀先验 P(type_k) = 1/num_task_types。
                                         可以是一个标量（所有类型相同）或一个向量。
        """
        self.agent_ids = list(agent_ids)
        self.task_ids = list(task_ids)
        self.num_task_types = num_task_types

        # beliefs 结构: {agent_id: {task_id: np.ndarray_alpha_values}}
        # np.ndarray_alpha_values 的形状是 (num_task_types,)
        self.beliefs: Dict[str, Dict[str, np.ndarray]] = {}

        if isinstance(initial_belief_alpha, (int, float)):
            # 如果是标量，为每种类型创建相同的初始伪计数
            alpha_vector = np.full(num_task_types, float(initial_belief_alpha))
        elif isinstance(initial_belief_alpha, (list, np.ndarray)):
            if len(initial_belief_alpha) != num_task_types:
                raise ValueError(f"初始伪计数向量的长度 ({len(initial_belief_alpha)}) "
                                 f"必须等于任务类型数量 ({num_task_types})。")
            alpha_vector = np.array(initial_belief_alpha, dtype=float)
        else:
            raise TypeError("initial_belief_alpha 必须是数字或向量。")

        for agent_id in self.agent_ids:
            self.beliefs[agent_id] = {}
            for task_id in self.task_ids:
                self.beliefs[agent_id][task_id] = alpha_vector.copy()

    def get_agent_belief_alphas_for_task(self, agent_id: str, task_id: str) -> Optional[np.ndarray]:
        """
        获取指定航天器对指定任务的当前信念伪计数 (alpha参数)。
        """
        if agent_id not in self.beliefs or task_id not in self.beliefs[agent_id]:
            # print(f"警告: 未找到航天器 {agent_id} 对任务 {task_id} 的信念记录。")
            return None
        return self.beliefs[agent_id][task_id].copy() # 返回副本以防外部修改

    def get_expected_belief_dist_for_task(self, agent_id: str, task_id: str) -> Optional[np.ndarray]:
        """
        计算指定航天器对指定任务的期望信念概率分布。
        P(type_k) = alpha_k / sum(alphas)
        """
        alphas = self.get_agent_belief_alphas_for_task(agent_id, task_id)
        if alphas is None:
            return None
        
        sum_alphas = np.sum(alphas)
        if sum_alphas <= 1e-9: # 避免除以零或非常小的值，理论上alpha应该为正
            # print(f"警告: Agent {agent_id}, Task {task_id} 的信念伪计数总和过小 ({sum_alphas})。返回均匀分布。")
            return np.full(self.num_task_types, 1.0 / self.num_task_types) # 返回均匀分布
        return alphas / sum_alphas

    def update_agent_belief_from_local_observation(self,
                                                 agent_id: str,
                                                 task_id: str,
                                                 observed_type_index: int,
                                                 observation_count: int = 1):
        """
        根据单个航天器的本地观测更新其对特定任务的信念。
        直接增加对应观测类型的伪计数。

        参数:
            agent_id (str): 进行观测的航天器ID。
            task_id (str): 被观测的任务ID。
            observed_type_index (int): 观测到的任务类型的索引 (0 到 num_task_types-1)。
            observation_count (int): 本次观测的计数（通常为1，但可以处理批量观测）。
        """
        if agent_id not in self.beliefs or task_id not in self.beliefs[agent_id]:
            print(f"警告: 尝试更新不存在的信念记录 (Agent: {agent_id}, Task: {task_id})。")
            return
        if not (0 <= observed_type_index < self.num_task_types):
            print(f"警告: 无效的观测类型索引 {observed_type_index}。")
            return
        if observation_count < 0:
            print(f"警告: 观测计数不能为负 {observation_count}。")
            return

        self.beliefs[agent_id][task_id][observed_type_index] += observation_count

    def aggregate_and_update_beliefs_from_shared_observations(self,
        shared_observations: Dict[str, np.ndarray]):
        """
        根据所有航天器共享的对各个任务的观测结果（已聚合），更新所有航天器的信念。
        这是实现“自学习”和全局信息融合的关键步骤。

        参数:
            shared_observations (Dict[str, np.ndarray]): 共享的观测结果。
                键是 task_id。
                值是一个形状为 (num_task_types,) 的NumPy数组，表示该任务在所有航天器中
                被观测为各个类型的总新增次数。
        """
        for task_id, task_observation_counts in shared_observations.items():
            if task_id not in self.task_ids:
                # print(f"警告: 收到未知任务 {task_id} 的共享观测，已忽略。")
                continue
            if not isinstance(task_observation_counts, np.ndarray) or task_observation_counts.shape != (self.num_task_types,):
                print(f"警告: 任务 {task_id} 的共享观测计数格式不正确 "
                      f"(期望形状 ({self.num_task_types},), 得到 {type(task_observation_counts)}, {task_observation_counts.shape if isinstance(task_observation_counts, np.ndarray) else 'N/A'})，已忽略。")
                continue

            for agent_id in self.agent_ids:
                if task_id in self.beliefs[agent_id]:
                    self.beliefs[agent_id][task_id] += task_observation_counts
                # else:
                    # print(f"警告: 航天器 {agent_id} 没有任务 {task_id} 的信念记录，无法更新共享观测。")

    def override_agent_belief_alphas_for_task(self,
                                            agent_id: str,
                                            task_id: str,
                                            new_alpha_values: np.ndarray):
        """
        直接覆盖指定航天器对指定任务的信念伪计数。
        用于设置特定的初始信念或在仿真中注入先验知识。

        参数:
            agent_id (str): 航天器ID。
            task_id (str): 任务ID。
            new_alpha_values (np.ndarray): 新的伪计数向量，形状应为 (num_task_types,)。
        """
        if agent_id not in self.beliefs:
            print(f"警告: 尝试覆盖未知航天器 {agent_id} 的信念，已忽略。")
            return
        if task_id not in self.beliefs[agent_id]:
            print(f"警告: 航天器 {agent_id} 没有任务 {task_id} 的信念记录，无法覆盖。")
            # 或者，如果希望在这种情况下创建记录：
            # self.beliefs[agent_id][task_id] = np.zeros(self.num_task_types)
            return

        if not isinstance(new_alpha_values, np.ndarray) or new_alpha_values.shape != (self.num_task_types,):
            print(f"警告: 为 Agent {agent_id}, Task {task_id} 提供的新伪计数形状不正确 "
                  f"(期望 ({self.num_task_types},), 得到 {new_alpha_values.shape})，已忽略。")
            return
        if np.any(new_alpha_values < 0): # 狄利克雷参数通常为正，至少是0
            print(f"警告: 为 Agent {agent_id}, Task {task_id} 提供的新伪计数包含负值，这可能不符合狄利克雷分布的常规用法。")


        self.beliefs[agent_id][task_id] = new_alpha_values.copy()


    def get_all_beliefs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """ 返回所有信念的深拷贝 """
        # Primitive deepcopy
        copied_beliefs = {}
        for agent_id, task_beliefs in self.beliefs.items():
            copied_beliefs[agent_id] = {}
            for task_id, alphas in task_beliefs.items():
                copied_beliefs[agent_id][task_id] = alphas.copy()
        return copied_beliefs
        

# --- 模块自测试与可视化 ---
def test_belief_manager():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 或者其他支持中文的字体
    matplotlib.rcParams['axes.unicode_minus'] = False

    agent_ids_test = ["sc1", "sc2"]
    task_ids_test = ["taskA", "taskB"]
    num_types_test = 3
    task_type_names_test = ["进攻型", "探测型", "维修型"] # 用于绘图标签

    # 1. 初始化 BeliefManager
    belief_mgr = BeliefManager(agent_ids_test, task_ids_test, num_types_test, initial_belief_alpha=0.1) # 先用一个小的均匀先验初始化

    # 使用新方法覆盖初始信念
    initial_alphas_sc1_taskA = np.array([1.0, 3.0, 1.0])
    initial_alphas_sc1_taskB = np.array([2.0, 1.0, 2.0])
    initial_alphas_sc2_taskA = np.array([3.0, 1.0, 1.0])
    initial_alphas_sc2_taskB = np.array([1.0, 1.0, 3.0])

    belief_mgr.override_agent_belief_alphas_for_task("sc1", "taskA", initial_alphas_sc1_taskA)
    belief_mgr.override_agent_belief_alphas_for_task("sc1", "taskB", initial_alphas_sc1_taskB)
    belief_mgr.override_agent_belief_alphas_for_task("sc2", "taskA", initial_alphas_sc2_taskA)
    belief_mgr.override_agent_belief_alphas_for_task("sc2", "taskB", initial_alphas_sc2_taskB)


    print("--- 初始信念 (期望概率) ---")
    for agent_id in agent_ids_test:
        for task_id in task_ids_test:
            dist = belief_mgr.get_expected_belief_dist_for_task(agent_id, task_id)
            print(f"  {agent_id} 对 {task_id}: {np.array2string(dist, precision=2, floatmode='fixed') if dist is not None else 'N/A'}")

    # 2. 模拟本地观测更新
    print("\n--- sc1 本地观测 taskA 为类型 0 (进攻型) 2次 ---")
    belief_mgr.update_agent_belief_from_local_observation("sc1", "taskA", 0, observation_count=2)
    dist_sc1_taskA_after_local = belief_mgr.get_expected_belief_dist_for_task("sc1", "taskA")
    print(f"  sc1 对 taskA 新信念: {np.array2string(dist_sc1_taskA_after_local, precision=2, floatmode='fixed') if dist_sc1_taskA_after_local is not None else 'N/A'}")
    alphas_sc1_taskA_after_local = belief_mgr.get_agent_belief_alphas_for_task("sc1", "taskA")
    print(f"  sc1 对 taskA 新伪计数: {alphas_sc1_taskA_after_local}")

    # 3. 模拟共享观测并聚合更新
    shared_obs_round1 = {
        "taskA": np.array([0, 3, 1]), 
        "taskB": np.array([2, 0, 0]) 
    }
    print(f"\n--- 共享观测并聚合更新 (第1轮) ---")
    print(f"  共享的观测增量: taskA: {shared_obs_round1['taskA']}, taskB: {shared_obs_round1['taskB']}")
    belief_mgr.aggregate_and_update_beliefs_from_shared_observations(shared_obs_round1)

    print("\n--- 信念更新后 (期望概率, 第1轮共享后) ---")
    beliefs_after_shared_r1 = {}
    for agent_id in agent_ids_test:
        beliefs_after_shared_r1[agent_id] = {}
        for task_id in task_ids_test:
            dist = belief_mgr.get_expected_belief_dist_for_task(agent_id, task_id)
            beliefs_after_shared_r1[agent_id][task_id] = dist
            print(f"  {agent_id} 对 {task_id}: {np.array2string(dist, precision=2, floatmode='fixed') if dist is not None else 'N/A'}")
            print(f"    伪计数: {belief_mgr.get_agent_belief_alphas_for_task(agent_id, task_id)}")
    
    # 4. 再进行一轮共享观测
    shared_obs_round2 = {
        "taskA": np.array([5, 1, 0]), 
        "taskB": np.array([0, 0, 4])  
    }
    print(f"\n--- 共享观测并聚合更新 (第2轮) ---")
    print(f"  共享的观测增量: taskA: {shared_obs_round2['taskA']}, taskB: {shared_obs_round2['taskB']}")
    belief_mgr.aggregate_and_update_beliefs_from_shared_observations(shared_obs_round2)
    
    print("\n--- 信念更新后 (期望概率, 第2轮共享后) ---")
    beliefs_after_shared_r2 = {}
    for agent_id in agent_ids_test:
        beliefs_after_shared_r2[agent_id] = {}
        for task_id in task_ids_test:
            dist = belief_mgr.get_expected_belief_dist_for_task(agent_id, task_id)
            beliefs_after_shared_r2[agent_id][task_id] = dist
            print(f"  {agent_id} 对 {task_id}: {np.array2string(dist, precision=2, floatmode='fixed') if dist is not None else 'N/A'}")
            print(f"    伪计数: {belief_mgr.get_agent_belief_alphas_for_task(agent_id, task_id)}")

    belief_history_sc1_taskA = []
    initial_dist_sc1_taskA = initial_alphas_sc1_taskA / np.sum(initial_alphas_sc1_taskA) if np.sum(initial_alphas_sc1_taskA) > 0 else np.full(num_types_test, 1.0/num_types_test)
    belief_history_sc1_taskA.append(initial_dist_sc1_taskA)
    if dist_sc1_taskA_after_local is not None: belief_history_sc1_taskA.append(dist_sc1_taskA_after_local)
    if "sc1" in beliefs_after_shared_r1 and "taskA" in beliefs_after_shared_r1["sc1"] and beliefs_after_shared_r1["sc1"]["taskA"] is not None:
        belief_history_sc1_taskA.append(beliefs_after_shared_r1["sc1"]["taskA"])
    if "sc1" in beliefs_after_shared_r2 and "taskA" in beliefs_after_shared_r2["sc1"] and beliefs_after_shared_r2["sc1"]["taskA"] is not None:
        belief_history_sc1_taskA.append(beliefs_after_shared_r2["sc1"]["taskA"])
    
    belief_history_sc1_taskA_np = np.array(belief_history_sc1_taskA) 

    stages = ["初始", "sc1本地观测后", "第1轮共享更新", "第2轮共享更新"]
    # 确保 stages 长度与 belief_history_sc1_taskA_np 的第一维匹配
    stages_plot = stages[:belief_history_sc1_taskA_np.shape[0]]


    if belief_history_sc1_taskA_np.size > 0 : # 确保有数据可画
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_types_test):
            ax.plot(stages_plot, belief_history_sc1_taskA_np[:, i], marker='o', linestyle='-', label=f'{task_type_names_test[i]} (TaskA)')
        
        ax.set_title(f'航天器 sc1 对任务 TaskA 的信念演化过程')
        ax.set_xlabel('信念更新阶段')
        ax.set_ylabel('期望概率')
        ax.legend(title='任务类型')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1) 
        plt.xticks(rotation=15) 
        plt.tight_layout()
        plt.show()
    else:
        print("无有效的信念历史数据可供绘制。")

if __name__ == '__main__':
    test_belief_manager()