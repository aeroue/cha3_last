# weak_communication/aif_functions_wrapper.py
import sys
import os

# --- 标准路径处理代码，确保独立运行时能找到项目模块 ---
def _add_project_root_to_sys_path():
    """将项目根目录添加到sys.path，以便模块导入。"""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir) # common, strong_communication 在 project_root 下面
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"调试：已将 '{project_root}' 添加到 sys.path (来自 ksc_coalition_formation.py)")
_add_project_root_to_sys_path()
# --- 路径处理代码结束 ---


import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import copy

# --- 临时路径处理 ---
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
_actual_project_root_directory = os.path.dirname(_project_root_directory)
if _actual_project_root_directory not in sys.path:
    sys.path.insert(0, _actual_project_root_directory)
# --- 结束 ---

# 导入我们刚刚完善的核心AIF函数模块
import weak_communication.aif_functions_isobeliefs_convergent as aif_core

# (如果 SpectralAnalyzer 在 common 包中，并且 common/__init__.py 存在)
# from common.spectral_analysis import SpectralAnalyzer # ActiveInferenceAgent 会用到它

class AIFFunctionsWrapper:
    """
    封装 aif_functions_isobeliefs_convergent.py 中的核心主动推理函数，
    为上层 ActiveInferenceAgent 提供更易用的接口。
    处理智能体ID映射、参数管理、状态观测转换等。
    """
    def __init__(self,
                 all_agent_ids_ordered: List[str], # 外部使用的字符串ID列表，顺序固定
                 num_goals: int,
                 all_agents_physical_params: List[Dict], # 每个agent的物理参数 (如type, capabilities)
                                                        # 顺序与 all_agent_ids_ordered 一致
                 goal_positions_list: List[np.ndarray], # 目标的物理位置列表
                 global_aif_hyperparams: Dict, # 全局AIF超参数 (如alpha, beta, gamma, policy_temp等)
                 allow_multiple_to_same_goal: bool = False
                 ):
        """
        初始化AIF封装器。

        参数:
            all_agent_ids_ordered: 系统中所有智能体的有序ID列表 (字符串)。
            num_goals: 系统中目标的总数。
            all_agents_physical_params: 包含每个智能体物理属性（如类型）的字典列表。
                                       列表顺序必须与 all_agent_ids_ordered 一致。
            goal_positions_list: 包含每个目标物理位置 (np.ndarray [x,y]) 的列表。
            global_aif_hyperparams: 包含AIF算法全局超参数的字典。
            allow_multiple_to_same_goal: 是否允许在目标配置G中将多个agent分配给同一个目标。
        """
        self.all_agent_ids_ordered = list(all_agent_ids_ordered)
        self.num_agents = len(all_agent_ids_ordered)
        self.agent_id_to_idx_map: Dict[str, int] = {agent_id: i for i, agent_id in enumerate(all_agent_ids_ordered)}
        self.idx_to_agent_id_map: Dict[int, str] = {i: agent_id for i, agent_id in enumerate(all_agent_ids_ordered)}

        if len(all_agents_physical_params) != self.num_agents:
            raise ValueError("all_agents_physical_params 列表长度必须与 all_agent_ids_ordered 一致。")

        # 初始化所有智能体的AIF核心参数
        self.agent_params_list_all: List[Dict] = []
        for i in range(self.num_agents):
            agent_id_str = self.all_agent_ids_ordered[i]
            physical_params = all_agents_physical_params[i]
            core_params = aif_core.initialize_agent_aif_params(
                agent_id_str=agent_id_str,
                agent_idx_internal=i,
                all_agent_ids_ordered=self.all_agent_ids_ordered,
                num_goals=num_goals,
                agent_physical_params=physical_params,
                goal_positions_list=goal_positions_list,
                global_aif_config=global_aif_hyperparams,
                allow_multiple_to_same_goal=allow_multiple_to_same_goal
            )
            self.agent_params_list_all.append(core_params)
        
        print(f"[AIFFunctionsWrapper] 已为 {self.num_agents} 个智能体初始化AIF参数。")
        if self.num_agents > 0 and self.agent_params_list_all[0]["possible_goal_configs_G"].size == 0 :
             print(f"警告: 可能的目标配置G为空。请检查 num_agents ({self.num_agents}), num_goals ({num_goals}), "
                   f"以及 allow_multiple_to_same_goal ({allow_multiple_to_same_goal})的设置。")


    def get_internal_idx(self, agent_id_str: str) -> Optional[int]:
        return self.agent_id_to_idx_map.get(agent_id_str)

    def get_agent_params(self, agent_id_str_or_idx: Any) -> Optional[Dict]:
        """获取指定agent的AIF核心参数字典"""
        if isinstance(agent_id_str_or_idx, str):
            idx = self.get_internal_idx(agent_id_str_or_idx)
            if idx is None: return None
            return self.agent_params_list_all[idx]
        elif isinstance(agent_id_str_or_idx, int):
            if 0 <= agent_id_str_or_idx < self.num_agents:
                return self.agent_params_list_all[agent_id_str_or_idx]
        return None

    def update_reasoning_mode_for_agent(self, agent_id_str: str, reasoning_level: int, use_ep: bool):
        """
        更新特定智能体的推理模式。
        注意：改变推理模式可能需要重置或调整其当前信念。
        """
        idx = self.get_internal_idx(agent_id_str)
        if idx is not None:
            self.agent_params_list_all[idx]["reasoning_level"] = reasoning_level
            self.agent_params_list_all[idx]["use_epistemic_planning"] = use_ep
            # print(f"  Agent {agent_id_str} (idx {idx}) 推理模式更新: level={reasoning_level}, use_ep={use_ep}")
            # 可选：重置其信念 q_G_posterior 为 q_G_prior，因为推理方式变了，旧后验可能无效。
            # num_configs = self.agent_params_list_all[idx]["possible_goal_configs_G"].shape[0] if self.agent_params_list_all[idx]["possible_goal_configs_G"].size >0 else 0
            # if num_configs > 0 :
            #    self.agent_params_list_all[idx]["q_G_posterior"] = np.ones(num_configs) / num_configs
            # else:
            #    self.agent_params_list_all[idx]["q_G_posterior"] = np.array([])

        # else:
            # print(f"警告: 尝试更新未知agent {agent_id_str} 的推理模式。")


    def update_beliefs_for_agent(self, 
                                 agent_id_str_deciding: str, # 当前做决策和更新信念的agent
                                 all_agent_true_states_global: List[np.ndarray] # 所有agent的当前真实物理状态列表
                                 ) -> Optional[np.ndarray]:
        """
        为指定的agent生成观测，计算似然，并更新其内部信念Q(G)。
        返回更新后的后验信念 Q(G)。
        """
        deciding_agent_idx = self.get_internal_idx(agent_id_str_deciding)
        if deciding_agent_idx is None:
            # print(f"错误: update_beliefs_for_agent 未找到agent_id {agent_id_str_deciding}")
            return None
        
        deciding_agent_params = self.agent_params_list_all[deciding_agent_idx]

        # 1. 为当前决策agent生成其观测
        #    current_observations_for_deciding_agent: Dict[internal_target_idx, observed_state]
        current_observations_for_deciding_agent = {}
        for target_j_idx in range(self.num_agents): # 迭代所有可能的被观测对象
            current_observations_for_deciding_agent[target_j_idx] = aif_core.simulate_observation_for_agent(
                agent_idx_observer=deciding_agent_idx, # 观测者是决策agent
                agent_idx_target=target_j_idx,         # 观测目标是j
                all_agent_true_states=all_agent_true_states_global, # 基于全局真实状态
                agent_params_observer=deciding_agent_params # 使用决策agent的观测模型
            )
        
        # 2. 计算似然 log L(G | current_observations)
        #    在高阶推理时，需要传入 all_agent_true_states_global作为 true_states_for_simulation
        true_states_for_higher_order_sim = all_agent_true_states_global \
            if deciding_agent_params["use_epistemic_planning"] and deciding_agent_params["reasoning_level"] >=2 \
            else None

        log_likelihood_G = aif_core.get_likelihood_for_agent(
            current_agent_idx=deciding_agent_idx,
            current_agent_observations=current_observations_for_deciding_agent,
            agent_params_list_all=self.agent_params_list_all, # 传递所有agent的参数
            true_states_for_simulation=true_states_for_higher_order_sim
        )

        if log_likelihood_G.size == 0: # 如果没有有效的G配置
            # print(f"警告: Agent {agent_id_str_deciding} 似然计算返回空，无法更新信念。")
            return deciding_agent_params["q_G_posterior"] # 返回现有信念

        # 3. 更新信念
        aif_core.update_belief(
            agent_idx=deciding_agent_idx,
            log_likelihood_G=log_likelihood_G,
            agent_params_list_all=self.agent_params_list_all # update_belief会修改此列表中的元素
        )
        
        return self.agent_params_list_all[deciding_agent_idx]["q_G_posterior"]


    def choose_action_for_agent(self,
                                agent_id_str_deciding: str,
                                candidate_actions_delta_s: List[np.ndarray], # 由谱分析提供
                                all_agent_true_states_global: List[np.ndarray]
                               ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """
        为指定agent从候选动作中选择一个最优动作（最小化EFE）。
        返回: (选择的delta_s, 对应的EFE值, 所有候选动作的EFE值列表)
        """
        deciding_agent_idx = self.get_internal_idx(agent_id_str_deciding)
        if deciding_agent_idx is None:
            # print(f"错误: choose_action_for_agent 未找到agent_id {agent_id_str_deciding}")
            return None, np.inf, None
            
        current_agent_physical_state = all_agent_true_states_global[deciding_agent_idx]

        selected_ds, min_efe, all_efes = aif_core.choice_heuristic(
            agent_idx=deciding_agent_idx,
            candidate_actions_delta_s=candidate_actions_delta_s,
            current_agent_physical_state=current_agent_physical_state,
            all_agent_current_states_for_sim=all_agent_true_states_global, # 用于EFE内部的未来状态模拟
            agent_params_list_all=self.agent_params_list_all
        )
        return selected_ds, min_efe, all_efes

# --- 模块自测试 (概念性) ---
def test_aif_wrapper():
    print("--- 测试 AIFFunctionsWrapper ---")
    num_agents_test = 2
    num_goals_test = 2
    agent_ids_str = [f"SC{i}" for i in range(num_agents_test)]
    
    # 异构的物理参数
    agent_phys_params = [
        {"type": 's', "agent_radius": 0.5, "evidence_sensitivity_eta": 30.0}, # Agent 0 (UGV-like)
        {"type": 'A', "agent_radius": 0.3, "evidence_sensitivity_eta": 20.0, "angle_evidence_weight_factor": 1.0}  # Agent 1 (UAV-like, angle primary)
    ]
    goal_positions = [np.array([50.0, 0.0]), np.array([-50.0, 0.0])]
    
    # 全局AIF超参数，控制推理级别等
    global_hyperparams = {
        "reasoning_level": 2, # 尝试二阶
        "use_epistemic_planning": True,
        "alpha_epistemic": 1.0, "beta_pragmatic": 0.7, "gamma_control_effort": 0.01,
        "policy_temperature": 0.05,
        "C_target_dist_G": np.array([0.8, 0.2]) # 假设G=[0,1]是0.8, G=[1,0]是0.2 (如果G配置只有这两种)
                                              # 这个需要根据 generate_possible_goal_configs 的结果来设定
    }

    # 确保 C_target_dist_G 长度正确
    _possible_G_wrapper_test = aif_core.generate_possible_goal_configs(num_agents_test, num_goals_test, False)
    if _possible_G_wrapper_test.size > 0:
        _num_configs_wrapper_test = _possible_G_wrapper_test.shape[0]
        if _num_configs_wrapper_test == 2: # 特指 2 agent, 2 goal, no repeat
             global_hyperparams["C_target_dist_G"] = np.array([0.8, 0.2])
        else:
             global_hyperparams["C_target_dist_G"] = np.ones(_num_configs_wrapper_test) / _num_configs_wrapper_test


    # 1. 初始化 Wrapper
    try:
        wrapper = AIFFunctionsWrapper(
            all_agent_ids_ordered=agent_ids_str,
            num_goals=num_goals_test,
            all_agents_physical_params=agent_phys_params,
            goal_positions_list=goal_positions,
            global_aif_hyperparams=global_hyperparams,
            allow_multiple_to_same_goal=False
        )
        print("AIFFunctionsWrapper 初始化成功。")
    except Exception as e:
        print(f"AIFFunctionsWrapper 初始化失败: {e}")
        return

    if wrapper.agent_params_list_all[0]["possible_goal_configs_G"].size == 0:
        print("没有可能的目标配置G，测试无法继续。")
        return

    # 2. 模拟全局状态
    current_true_states = [
        np.array([45.0, 2.0, 0.5, 0.0]), # SC0 接近 goal0
        np.array([-48.0, -1.0, -0.5, 0.0]) # SC1 接近 goal1
    ]

    # 3. 测试 SC0 的信念更新
    print(f"\n--- 测试 {agent_ids_str[0]} 的信念更新 ---")
    q_G_posterior_sc0_before = wrapper.get_agent_params(agent_ids_str[0])["q_G_posterior"].copy()
    print(f"  {agent_ids_str[0]} 更新前信念 Q(G): {q_G_posterior_sc0_before}")
    
    updated_q_G_sc0 = wrapper.update_beliefs_for_agent(agent_ids_str[0], current_true_states)
    if updated_q_G_sc0 is not None:
        print(f"  {agent_ids_str[0]} 更新后信念 Q(G): {updated_q_G_sc0}")
        # 验证信念是否真的改变了 (由于随机观测，通常会变)
        if not np.allclose(q_G_posterior_sc0_before, updated_q_G_sc0):
            print("    信念已成功更新。")
        else:
            print("    警告：信念未发生显著变化。")

    # 4. 测试 SC0 的动作选择
    print(f"\n--- 测试 {agent_ids_str[0]} 的动作选择 ---")
    # 假设的候选动作 (状态偏差)
    candidate_actions = [
        np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]) # 零动作
    ]
    selected_action, efe_value, all_efes = wrapper.choose_action_for_agent(
        agent_ids_str[0], candidate_actions, current_true_states
    )
    if selected_action is not None:
        print(f"  {agent_ids_str[0]} 选择的动作 (delta_s): {selected_action[:2]}, EFE: {efe_value:.3f}")
        print(f"  所有动作的EFE: {all_efes}")
    else:
        print(f"  {agent_ids_str[0]} 未能选择动作。")
        
    # 5. 测试更改推理模式
    print(f"\n--- 测试更改 {agent_ids_str[1]} 的推理模式 ---")
    params_sc1_before_mode_change = copy.deepcopy(wrapper.get_agent_params(agent_ids_str[1]))
    wrapper.update_reasoning_mode_for_agent(agent_ids_str[1], reasoning_level=0, use_ep=False) # 改为零阶
    params_sc1_after_mode_change = wrapper.get_agent_params(agent_ids_str[1])
    print(f"  {agent_ids_str[1]} 修改前: level={params_sc1_before_mode_change['reasoning_level']}, use_ep={params_sc1_before_mode_change['use_epistemic_planning']}")
    print(f"  {agent_ids_str[1]} 修改后: level={params_sc1_after_mode_change['reasoning_level']}, use_ep={params_sc1_after_mode_change['use_epistemic_planning']}")
    if params_sc1_after_mode_change['reasoning_level'] == 0 and not params_sc1_after_mode_change['use_epistemic_planning']:
        print("    推理模式已成功更改为零阶。")
    else:
        print("    警告：推理模式更改未按预期生效。")


if __name__ == '__main__':
    test_aif_wrapper()