# weak_communication/aif_functions_isobeliefs_convergent.py

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
import copy
import itertools
from scipy.stats import dirichlet as dirichlet_dist # 重命名以避免与参数名冲突
from typing import List, Dict, Tuple, Any, Optional, Callable # 确保导入了 Callable
import math

def initialize_agent_aif_params(agent_id_str: str, agent_idx_internal: int,
                                all_agent_ids_ordered: List[str], num_goals: int,
                                agent_physical_params: Dict, 
                                goal_positions_list: List[np.ndarray],
                                global_aif_config: Dict, 
                                allow_multiple_to_same_goal: bool = False
                               ) -> Dict[str, Any]:
    num_agents = len(all_agent_ids_ordered)
    params = copy.deepcopy(DEFAULT_AIF_PARAMS) 
    params.update(global_aif_config) 
    
    # 先将 agent_physical_params 中与 DEFAULT_AIF_PARAMS 重叠的键更新到 params
    # 这样 agent_physical_params 中的特定设置可以覆盖全局或默认
    for key in agent_physical_params:
        if key in params: # 如果是AIF已知参数
            params[key] = agent_physical_params[key]
        # else: # 如果是agent特有的物理属性，不直接属于AIF核心参数，但可能被custom_evidence等使用
        #     params[key] = agent_physical_params[key] # 也可以选择性加入

    # 明确处理 agent_type
    if 'type' in agent_physical_params:
        params["agent_type"] = agent_physical_params['type']
    elif 'agent_type' in agent_physical_params: # 也检查下划线形式
        params["agent_type"] = agent_physical_params['agent_type']
    else:
        params["agent_type"] = 'default_generic_type' # 如果没有提供，则设置一个默认类型
        print(f"警告: Agent {agent_id_str} 未在 physical_params 中提供 'type'，使用默认值 '{params['agent_type']}'。")


    params["agent_id_str"] = agent_id_str
    params["agent_idx_internal"] = agent_idx_internal
    params["num_agents"] = num_agents
    params["num_goals"] = num_goals
    params["all_agent_ids_ordered"] = list(all_agent_ids_ordered) 

    params["goals_phys_pos"] = np.array(goal_positions_list) 
    params["goals_phys_pos_dict"] = {idx: pos for idx, pos in enumerate(goal_positions_list)}

    possible_G_configs_array = generate_possible_goal_configs(num_agents, num_goals, allow_multiple_to_same_goal)
    params["possible_goal_configs_G"] = possible_G_configs_array
    
    num_total_configs = possible_G_configs_array.shape[0] if possible_G_configs_array.size > 0 else 0

    if num_total_configs > 0:
        q_G_prior = np.ones(num_total_configs) / num_total_configs
        params["q_G_prior"] = q_G_prior
        params["q_G_posterior"] = q_G_prior.copy()
        
        # C_target_dist_G: 如果全局配置里有，并且长度匹配，就用它
        c_target_global = params.get("C_target_dist_G")
        if c_target_global is not None and isinstance(c_target_global, np.ndarray) and len(c_target_global) == num_total_configs:
            params["C_target_dist_G"] = c_target_global.copy() # 使用全局的
        else: # 否则，默认均匀
            params["C_target_dist_G"] = q_G_prior.copy()

        if params.get("utility_of_G_configs") is None or \
           not isinstance(params.get("utility_of_G_configs"), np.ndarray) or \
           len(params.get("utility_of_G_configs")) != num_total_configs:
            params["utility_of_G_configs"] = np.zeros(num_total_configs)
        
        params["beliefs_about_others_q_G"] = [q_G_prior.copy() for _ in range(num_agents)]
    else:
        params["q_G_prior"] = np.array([])
        params["q_G_posterior"] = np.array([])
        params["C_target_dist_G"] = np.array([])
        params["utility_of_G_configs"] = np.array([])
        params["beliefs_about_others_q_G"] = []
    
    # 调试打印，确认 agent_type 是否存在
    # if 'agent_type' not in params:
    #     print(f"CRITICAL ERROR: 'agent_type' still not in params for {agent_id_str} after initialization attempt.")
    #     print(f"  physical_params received: {agent_physical_params}")
    #     print(f"  global_aif_config received: {global_aif_config}")
    #     print(f"  Final params dict: {params}")
    # else:
    #     print(f"  Agent {agent_id_str} initialized with type: {params['agent_type']}")

    return params

# ... (其余所有函数，包括 custom_evidence_calculation_from_original, compute_consensus_of_beliefs,
#      get_likelihood_for_agent, update_belief, calculate_expected_free_energy, choice_heuristic
#      以及测试函数 if __name__ == '__main__': test_aif_core_functions() 都保持与上一版本相同) ...
# --- 为了简洁，我将省略这些已定义的函数，直接跳到测试用例的调用部分 ---

# (将上一版本中从 def softmax(...) 到 def choice_heuristic(...) 的所有函数代码粘贴到这里)
# ... (所有函数定义) ...
# 确保您已将上一版本中的所有函数（softmax 到 choice_heuristic）复制到此处
# 我将直接提供 `if __name__ == '__main__':` 部分，假设以上函数已正确定义。

# === 1. 基础数学工具函数 (与上一版本相同) ===
def softmax(x_in: np.ndarray, t: float = 1.0) -> np.ndarray:
    if x_in.ndim == 1: x_in = x_in.reshape(1, -1)
    x_max = np.max(x_in, axis=1, keepdims=True)
    x_stable = (x_in - x_max) / t
    e_x = np.exp(x_stable)
    sum_e_x = np.sum(e_x, axis=1, keepdims=True)
    return e_x / (sum_e_x + 1e-12) 

def log_stable(x_in: np.ndarray, epsilon: float = 1e-32) -> np.ndarray:
    return np.log(np.maximum(x_in, epsilon)) 

def wrapToPi(x_in: float) -> float:
    return (x_in + np.pi) % (2 * np.pi) - np.pi

def entropy(prob_dist: np.ndarray) -> float:
    if prob_dist.ndim > 1: prob_dist = prob_dist.flatten()
    prob_dist = np.maximum(prob_dist, 1e-12) 
    prob_dist_sum = np.sum(prob_dist)
    if abs(prob_dist_sum - 1.0) > 1e-6 and prob_dist_sum > 1e-9:
        prob_dist = prob_dist / prob_dist_sum
    elif prob_dist_sum < 1e-9:
        if len(prob_dist) > 0: return -log_stable(1.0/len(prob_dist))
        else: return 0.0
    return -np.sum(prob_dist * log_stable(prob_dist))

def kl_divergence(p_dist: np.ndarray, q_dist: np.ndarray) -> float:
    if p_dist.ndim > 1: p_dist = p_dist.flatten()
    if q_dist.ndim > 1: q_dist = q_dist.flatten()
    p_dist = np.maximum(p_dist, 1e-12); p_dist_sum = np.sum(p_dist); p_dist = p_dist / (p_dist_sum if p_dist_sum > 1e-9 else 1.0)
    q_dist = np.maximum(q_dist, 1e-12); q_dist_sum = np.sum(q_dist); q_dist = q_dist / (q_dist_sum if q_dist_sum > 1e-9 else 1.0)
    return np.sum(p_dist * (log_stable(p_dist) - log_stable(q_dist)))

# === 2. 参数初始化与配置 (与上一版本相同) ===
DEFAULT_AIF_PARAMS = {
    "alpha_epistemic": 1.0, "beta_pragmatic": 1.0, "gamma_control_effort": 0.01,
    "obs_variance_pos": 0.5, "obs_variance_vel": 0.05, # 方差
    "reasoning_level": 0, "use_epistemic_planning": False,
    "perspective_fusion_lambda": 0.5, 
    "utility_of_G_configs": None, "C_target_dist_G": None,      
    "action_outcome_variance": 0.1, 
    "evidence_sensitivity_eta": 30.0, 
    "policy_temperature": 0.1, 
    "belief_update_temperature": 1.0,
    "planning_horizon_H_aif": 1,
    "agent_capabilities": [], 
    "evidence_types_for_capability": {"distance_sensor": ["distance_to_goal"], "angle_sensor": ["angle_to_goal"]},
    "safety_distance_sq": 25.0, 
    "collision_penalty_h_factor": 100.0,
    "vel_mag_thresh": 0.01, 
    "agent_radius": 0.5,   
    "dist_thresh_evidence": 50.0, 
    "angle_evidence_weight_factor": 1.0, 
    "evidence_norm_factor": 1.0, 
    "EPS": 1e-9 
}

def generate_possible_goal_configs(num_agents: int, num_goals: int, allow_multiple_to_same_goal: bool = False) -> np.ndarray:
    goal_indices = list(range(num_goals))
    if num_agents == 0 or num_goals == 0: return np.array([])
    if allow_multiple_to_same_goal:
        all_configs = list(itertools.product(goal_indices, repeat=num_agents))
    else:
        if num_agents > num_goals: return np.array([])
        all_configs = list(itertools.permutations(goal_indices, num_agents))
    return np.array(all_configs) if all_configs else np.array([])

# initialize_agent_aif_params (已在上面修正)

# === 3. 观测模型与证据计算 (custom_evidence_calculation 更新) ===
def simulate_observation_for_agent(agent_idx_observer: int, agent_idx_target: int,
                                   all_agent_true_states: List[np.ndarray], 
                                   agent_params_observer: Dict) -> np.ndarray:
    true_state_target = all_agent_true_states[agent_idx_target]
    obs_std_pos = np.sqrt(agent_params_observer.get("obs_variance_pos", DEFAULT_AIF_PARAMS["obs_variance_pos"]))
    obs_std_vel = np.sqrt(agent_params_observer.get("obs_variance_vel", DEFAULT_AIF_PARAMS["obs_variance_vel"]))
    obs_noise_pos = np.random.normal(0, obs_std_pos, 2) 
    obs_noise_vel = np.random.normal(0, obs_std_vel, 2) 
    observed_state = np.concatenate((true_state_target[:2] + obs_noise_pos, 
                                     true_state_target[2:] + obs_noise_vel))
    return observed_state

def custom_evidence_calculation_from_original( 
    agent_idx_internal: int, 
    agent_type: str,      
    agent_observed_state: np.ndarray, 
    target_goal_pos: np.ndarray,    
    agent_params: Dict, 
    all_agents_observed_states_dict: Optional[Dict[int, np.ndarray]] = None, 
    goal_config_vector_Gk: Optional[np.ndarray] = None, 
    evidence_meta_type: str = "self_to_goal",
    eta: float = 30.0 # 从 agent_params 获取或使用默认
    ) -> float:
    
    pos_agent = agent_observed_state[:2]
    vel_agent = agent_observed_state[2:]
    EPS = agent_params.get("EPS", DEFAULT_AIF_PARAMS["EPS"]) 

    diff_to_goal = target_goal_pos - pos_agent
    dist_sq_to_goal = np.sum(diff_to_goal**2)
    h_dist = dist_sq_to_goal 

    h_angle = 0.0
    # 使用 agent_params 中的 vel_mag_thresh
    vel_mag_thresh_local = agent_params.get("vel_mag_thresh", DEFAULT_AIF_PARAMS["vel_mag_thresh"])
    if agent_type == 'A' or agent_type == 'B':
        vel_magnitude = np.linalg.norm(vel_agent)
        dist_to_goal = np.sqrt(dist_sq_to_goal)

        if vel_magnitude > vel_mag_thresh_local and dist_to_goal > EPS:
            dir_vel_agent = vel_agent / (vel_magnitude + EPS)
            dir_to_goal = diff_to_goal / (dist_to_goal + EPS)
            cos_theta = np.dot(dir_vel_agent, dir_to_goal)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_mismatch_orig = np.abs(np.arccos(cos_theta) - (np.pi / 2.0)) 
            h_angle = angle_mismatch_orig 
    
    h_combined = 0.0
    norm_factor_local = agent_params.get("evidence_norm_factor", DEFAULT_AIF_PARAMS["evidence_norm_factor"])
    if agent_type == 's': 
        h_combined = h_dist / norm_factor_local 
    elif agent_type == 'A': 
        h_combined = h_angle * (agent_params.get("dist_thresh_evidence", 50.0)**2 / (np.pi/2)) / norm_factor_local 
    elif agent_type == 'B': 
        dist_weight_local = agent_params.get("dist_evidence_weight", 1.0) # 新增，如果需要
        angle_comb_weight_local = agent_params.get("angle_evidence_weight_factor", 50.0) 
        h_combined = (dist_weight_local * h_dist + angle_comb_weight_local * h_angle) / norm_factor_local
    else: 
        h_combined = h_dist / norm_factor_local
        
    h_collaborative = 0.0
    if all_agents_observed_states_dict and goal_config_vector_Gk is not None and \
       (evidence_meta_type == "self_to_goal"):
        my_goal_idx_in_Gk = -1
        for g_idx_find, g_pos_find in enumerate(agent_params["goals_phys_pos"]):
            if np.array_equal(g_pos_find, target_goal_pos): 
                my_goal_idx_in_Gk = g_idx_find; break
        if my_goal_idx_in_Gk != -1:
            for other_idx, other_state_obs in all_agents_observed_states_dict.items():
                if other_idx == agent_idx_internal: continue
                if goal_config_vector_Gk[other_idx] == my_goal_idx_in_Gk:
                    dist_to_other_sq = np.sum((agent_observed_state[:2] - other_state_obs[:2])**2)
                    if dist_to_other_sq < agent_params.get("safety_distance_sq", DEFAULT_AIF_PARAMS["safety_distance_sq"]):
                        h_collaborative += agent_params.get("collision_penalty_h_factor", DEFAULT_AIF_PARAMS["collision_penalty_h_factor"])
    h_combined += h_collaborative
    
    return h_combined # 注意：原始custom_cdist返回的是概率，这里返回h。eta在外部用。


def calculate_log_evidence_contribution( 
    perspective_agent_idx: int, perspective_agent_params: Dict,
    evidence_for_agent_idx: int, 
    observed_state_of_evidence_agent: np.ndarray, 
    goal_config_vector_Gk: np.ndarray,
    agent_params_list_all: List[Dict],
    all_observed_states_from_perspective: Dict[int, np.ndarray] 
    ) -> float:
    
    assigned_goal_idx = goal_config_vector_Gk[evidence_for_agent_idx]
    if assigned_goal_idx < 0 or assigned_goal_idx >= len(perspective_agent_params["goals_phys_pos"]):
        return -1e12 

    target_pos_for_evidence_agent = perspective_agent_params["goals_phys_pos"][assigned_goal_idx]
    params_of_evidence_agent = agent_params_list_all[evidence_for_agent_idx]
    agent_type_of_evidence_agent = params_of_evidence_agent["agent_type"]
    eta_local = params_of_evidence_agent.get("evidence_sensitivity_eta", DEFAULT_AIF_PARAMS["evidence_sensitivity_eta"])
    if eta_local == 0 : eta_local = 1e-9

    h_value = custom_evidence_calculation_from_original( 
        agent_idx_internal=evidence_for_agent_idx, 
        agent_type=agent_type_of_evidence_agent,
        agent_observed_state=observed_state_of_evidence_agent,
        target_goal_pos=target_pos_for_evidence_agent, 
        agent_params=params_of_evidence_agent, 
        all_agents_observed_states_dict=all_observed_states_from_perspective, 
        goal_config_vector_Gk=goal_config_vector_Gk,
        evidence_meta_type="other_to_goal_from_my_obs" if evidence_for_agent_idx != perspective_agent_idx else "self_to_goal",
        eta=eta_local 
    )
    return -h_value / eta_local


def compute_consensus_of_beliefs( 
    list_of_prob_distributions_for_G: List[np.ndarray], 
    agent_types_for_perspectives: List[str], 
    method: str = 'entropy_weighted', 
    perspective_fusion_lambda: Optional[float] = None, 
    own_prob_distribution_for_G: Optional[np.ndarray] = None 
    ) -> np.ndarray:
    if not list_of_prob_distributions_for_G and own_prob_distribution_for_G is not None:
        return own_prob_distribution_for_G
    if not list_of_prob_distributions_for_G and own_prob_distribution_for_G is None:
        return np.array([])


    num_perspectives = len(list_of_prob_distributions_for_G)
    # 处理 own_prob_distribution_for_G 为空但 list_of_prob_distributions_for_G 不为空的情况
    if own_prob_distribution_for_G is None and num_perspectives > 0:
        # 如果自己的分布未提供，但有其他人的，可以只用其他人的，或报错
        # 这里我们简化为如果isobelief方法且own_prob缺失则退化
        if method == 'isobelief':
            # print("警告: Isobelief 方法需要 own_prob_distribution_for_G，但未提供。")
            pass # compute_consensus的isobelief分支会处理

    # 确保所有输入分布的形状一致 (如果列表不为空)
    if num_perspectives > 0:
        first_dist_shape = list_of_prob_distributions_for_G[0].shape
        if not all(dist.shape == first_dist_shape for dist in list_of_prob_distributions_for_G):
            # print("错误: compute_consensus 中输入的概率分布形状不一致。")
            # 尝试返回第一个作为回退，或一个空数组
            return list_of_prob_distributions_for_G[0] if num_perspectives >0 else np.array([])
        np_prob_distributions = np.array(list_of_prob_distributions_for_G)
    else: # list_of_prob_distributions_for_G 为空
        if own_prob_distribution_for_G is not None: return own_prob_distribution_for_G
        else: return np.array([])


    if method == 'simple_avg':
        fused_prob_dist = np.mean(np_prob_distributions, axis=0)
        return fused_prob_dist / (np.sum(fused_prob_dist) + 1e-9)

    elif method == 'entropy_weighted': 
        entropies_val = np.array([entropy(p_dist) for p_dist in np_prob_distributions])
        weights = 1.0 / (entropies_val + 1e-6) 
        if np.sum(weights) < 1e-9 : 
            weights = np.ones(num_perspectives) 
        weights = weights / np.sum(weights) 
        fused_prob_dist = np.sum(np_prob_distributions * weights[:, np.newaxis], axis=0)
        return fused_prob_dist / (np.sum(fused_prob_dist) + 1e-9)

    elif method == 'isobelief': 
        if own_prob_distribution_for_G is None or perspective_fusion_lambda is None:
            # print("警告: Isobelief 方法参数不足。退化为简单平均。")
            if num_perspectives > 0:
                fused_prob_dist = np.mean(np_prob_distributions, axis=0)
                return fused_prob_dist / (np.sum(fused_prob_dist) + 1e-9)
            return np.array([]) # 无法计算

        if num_perspectives > 0:
            avg_others_prob_dist = np.mean(np_prob_distributions, axis=0)
            fused_prob_dist = (1 - perspective_fusion_lambda) * own_prob_distribution_for_G + \
                               perspective_fusion_lambda * avg_others_prob_dist
            return fused_prob_dist / (np.sum(fused_prob_dist) + 1e-9)
        else:
            return own_prob_distribution_for_G # 没有其他视角，直接用自己的
    else: 
        if num_perspectives > 0:
            fused_prob_dist = np.mean(np_prob_distributions, axis=0)
            return fused_prob_dist / (np.sum(fused_prob_dist) + 1e-9)
        elif own_prob_distribution_for_G is not None:
            return own_prob_distribution_for_G
        return np.array([])


def get_likelihood_for_agent( 
    current_agent_idx: int, 
    current_agent_observations: Dict[int, np.ndarray], 
    agent_params_list_all: List[Dict], 
    true_states_for_simulation: Optional[List[np.ndarray]] = None 
    ) -> np.ndarray: # 返回对数似然向量
    perspective_agent_params = agent_params_list_all[current_agent_idx]
    possible_G_configs = perspective_agent_params["possible_goal_configs_G"]
    num_configs = possible_G_configs.shape[0] if possible_G_configs.ndim > 1 and possible_G_configs.size > 0 else 0
    if num_configs == 0: return np.array([])

    # 1. 计算当前视角agent自己的一阶对数似然 L_i(G|y_i) (这是一个向量，对所有Gk)
    first_order_log_L_vector_self = np.zeros(num_configs)
    for config_idx_inner, G_k_assignment_vector_inner in enumerate(possible_G_configs): # 修改循环变量名
        log_evidence_for_this_Gk_inner = 0.0
        for agent_j_target_idx_inner in range(perspective_agent_params["num_agents"]): # 修改循环变量名
            observed_state_of_j_inner = current_agent_observations.get(agent_j_target_idx_inner)
            if observed_state_of_j_inner is not None:
                log_evidence_for_this_Gk_inner += calculate_log_evidence_contribution(
                    current_agent_idx, perspective_agent_params,
                    agent_j_target_idx_inner, observed_state_of_j_inner,
                    G_k_assignment_vector_inner, agent_params_list_all, current_agent_observations
                )
            # else: 如果观测不到，可以加一个惩罚或忽略，当前是贡献为0 (即-h=0, h=0, exp(0)=1, 似然中性)
            # 或者应该是一个大的负值，表示非常不可能
        first_order_log_L_vector_self[config_idx_inner] = log_evidence_for_this_Gk_inner
    
    reasoning_level = perspective_agent_params.get("reasoning_level", 0)
    use_ep = perspective_agent_params.get("use_epistemic_planning", False)

    if not use_ep or reasoning_level < 2: 
        if reasoning_level == 0: 
            zeroth_order_log_L_vector = np.zeros(num_configs)
            for config_idx_z, G_k_assignment_vector_z in enumerate(possible_G_configs):
                my_observed_state_z = current_agent_observations.get(current_agent_idx)
                if my_observed_state_z is not None:
                    zeroth_order_log_L_vector[config_idx_z] = calculate_log_evidence_contribution(
                        current_agent_idx,perspective_agent_params, current_agent_idx, 
                        my_observed_state_z, G_k_assignment_vector_z, agent_params_list_all,
                        {current_agent_idx: my_observed_state_z} 
                    )
                else: zeroth_order_log_L_vector[config_idx_z] = -1e9
            return zeroth_order_log_L_vector
        else: 
            return first_order_log_L_vector_self
    else: # 高阶推理
        if true_states_for_simulation is None:
            return first_order_log_L_vector_self

        all_other_agents_simulated_prob_dist_vectors = [] 
        other_agent_types_for_consensus = []

        for other_agent_j_idx_ho in range(perspective_agent_params["num_agents"]): # 修改循环变量名
            if other_agent_j_idx_ho == current_agent_idx: continue 

            simulated_obs_for_j_by_i_ho = {} 
            for agent_k_target_for_j_obs_ho in range(perspective_agent_params["num_agents"]):
                 simulated_obs_for_j_by_i_ho[agent_k_target_for_j_obs_ho] = simulate_observation_for_agent(
                     other_agent_j_idx_ho, agent_k_target_for_j_obs_ho, 
                     true_states_for_simulation, agent_params_list_all[other_agent_j_idx_ho] 
                 )
            
            temp_params_for_j_sim_ho = copy.deepcopy(agent_params_list_all) 
            temp_params_for_j_sim_ho[other_agent_j_idx_ho]["reasoning_level"] = 1 
            temp_params_for_j_sim_ho[other_agent_j_idx_ho]["use_epistemic_planning"] = False 

            log_L_G_vector_from_j_perspective_ho = get_likelihood_for_agent(
                current_agent_idx=other_agent_j_idx_ho, 
                current_agent_observations=simulated_obs_for_j_by_i_ho, 
                agent_params_list_all=temp_params_for_j_sim_ho, 
                true_states_for_simulation=true_states_for_simulation 
            )
            # 转换为概率分布 P_j(G|y_j_sim)
            prob_dist_from_j = softmax(log_L_G_vector_from_j_perspective_ho, 
                                       t=temp_params_for_j_sim_ho[other_agent_j_idx_ho].get("belief_update_temperature", 1.0))
            all_other_agents_simulated_prob_dist_vectors.append(prob_dist_from_j.flatten())
            other_agent_types_for_consensus.append(agent_params_list_all[other_agent_j_idx_ho]["agent_type"])
        
        if all_other_agents_simulated_prob_dist_vectors:
            own_prob_dist_for_consensus = softmax(first_order_log_L_vector_self, 
                                                  t=perspective_agent_params.get("belief_update_temperature",1.0)).flatten()
            
            fused_prob_dist = compute_consensus_of_beliefs(
                list_of_prob_distributions_for_G=all_other_agents_simulated_prob_dist_vectors,
                agent_types_for_perspectives=other_agent_types_for_consensus, 
                method='isobelief', # 或 'entropy_weighted'
                perspective_fusion_lambda=perspective_agent_params["perspective_fusion_lambda"],
                own_prob_distribution_for_G=own_prob_dist_for_consensus
            )
            return log_stable(fused_prob_dist) # 返回融合后概率的对数
        else:
            return first_order_log_L_vector_self


def update_belief(agent_idx: int, log_likelihood_G: np.ndarray, agent_params_list_all: List[Dict]):
    # (与上一版本相同)
    my_params = agent_params_list_all[agent_idx]
    if my_params["q_G_prior"] is None or len(my_params["q_G_prior"]) != len(log_likelihood_G):
        num_configs = len(log_likelihood_G)
        my_params["q_G_prior"] = np.ones(num_configs) / num_configs if num_configs > 0 else np.array([])
    if len(my_params["q_G_prior"]) == 0:
        my_params["q_G_posterior"] = np.array([]); return
    log_q_G_posterior_unnormalized = log_likelihood_G + log_stable(my_params["q_G_prior"])
    belief_temp = my_params.get("belief_update_temperature", DEFAULT_AIF_PARAMS["belief_update_temperature"])
    my_params["q_G_posterior"] = softmax(log_q_G_posterior_unnormalized, t=belief_temp).flatten()


def calculate_expected_free_energy( # (与上一版本相同)
    agent_idx: int, candidate_action_delta_s: np.ndarray, 
    current_agent_physical_state: np.ndarray, 
    all_agent_current_states_for_sim: List[np.ndarray], 
    agent_params_list_all: List[Dict]
    ) -> float:
    # (与上一版本相同)
    my_params = agent_params_list_all[agent_idx]
    num_configs_G = my_params["possible_goal_configs_G"].shape[0] if my_params["possible_goal_configs_G"].size > 0 else 0
    if num_configs_G == 0: return np.inf
    s_prime_agent = current_agent_physical_state + candidate_action_delta_s
    future_all_agent_states_for_sim = [s.copy() for s in all_agent_current_states_for_sim]
    future_all_agent_states_for_sim[agent_idx] = s_prime_agent
    sim_y_prime_for_agent = {}
    for target_j_idx in range(my_params["num_agents"]):
        sim_y_prime_for_agent[target_j_idx] = simulate_observation_for_agent(
            agent_idx, target_j_idx, future_all_agent_states_for_sim, my_params)
    log_L_G_given_y_prime = get_likelihood_for_agent(
        agent_idx, sim_y_prime_for_agent, agent_params_list_all, future_all_agent_states_for_sim)
    q_G_prior_for_future_state = my_params["q_G_posterior"]
    if q_G_prior_for_future_state is None or len(q_G_prior_for_future_state) != num_configs_G:
        q_G_prior_for_future_state = my_params["q_G_prior"]
    if len(q_G_prior_for_future_state) != len(log_L_G_given_y_prime): 
        if len(log_L_G_given_y_prime) > 0 :
            q_G_prior_for_future_state = np.ones_like(log_L_G_given_y_prime) / len(log_L_G_given_y_prime)
        else: return np.inf
    log_q_G_posterior_prime_unnorm = log_L_G_given_y_prime + log_stable(q_G_prior_for_future_state)
    q_G_posterior_prime = softmax(log_q_G_posterior_prime_unnorm, 
                                  t=my_params.get("belief_update_temperature", 1.0)).flatten()
    epistemic_term = np.sum(q_G_posterior_prime * log_stable(q_G_posterior_prime)) 
    pragmatic_term = 0.0
    C_target = my_params.get("C_target_dist_G")
    U_of_G = my_params.get("utility_of_G_configs")
    if C_target is not None and len(C_target) == num_configs_G:
        pragmatic_term = -kl_divergence(q_G_posterior_prime, C_target) 
    elif U_of_G is not None and len(U_of_G) == num_configs_G:
        pragmatic_term = np.sum(q_G_posterior_prime * U_of_G)
    control_effort_cost = np.sum(candidate_action_delta_s**2)
    alpha = my_params["alpha_epistemic"]; beta = my_params["beta_pragmatic"]; gamma = my_params["gamma_control_effort"]
    efe = alpha * epistemic_term - beta * pragmatic_term + gamma * control_effort_cost
    return efe


def choice_heuristic( # (与上一版本相同)
    agent_idx: int, candidate_actions_delta_s: List[np.ndarray], 
    current_agent_physical_state: np.ndarray,
    all_agent_current_states_for_sim: List[np.ndarray],
    agent_params_list_all: List[Dict]
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
    # (与上一版本相同)
    # print(len(candidate_actions_delta_s))
    my_params = agent_params_list_all[agent_idx]
    if not candidate_actions_delta_s: return None, np.inf, None
    efes = np.array([calculate_expected_free_energy(
            agent_idx, action_ds, current_agent_physical_state,
            all_agent_current_states_for_sim, agent_params_list_all
        ) for action_ds in candidate_actions_delta_s])
    if np.all(np.isinf(efes)) or np.all(np.isnan(efes)) : 
        zero_action_idx = -1
        for idx_a, act_a in enumerate(candidate_actions_delta_s):
            if np.allclose(act_a, 0): zero_action_idx = idx_a; break
        if zero_action_idx != -1: return candidate_actions_delta_s[zero_action_idx], np.inf, efes
        return candidate_actions_delta_s[np.random.choice(len(candidate_actions_delta_s))] if candidate_actions_delta_s else None, np.inf, efes
    policy_temp = my_params.get("policy_temperature", DEFAULT_AIF_PARAMS["policy_temperature"])
    if abs(policy_temp) < 1e-9: 
        best_action_idx = np.nanargmin(efes) 
    else:
        log_probs_action = -efes / policy_temp 
        action_probs = softmax(log_probs_action)
        if action_probs.ndim > 1 : action_probs = action_probs.flatten()
        if np.any(np.isnan(action_probs)) or abs(np.sum(action_probs) - 1.0) > 1e-6 or np.sum(action_probs) < 1e-9 :
            best_action_idx = np.nanargmin(efes) 
        else:
            try: best_action_idx = np.random.choice(len(candidate_actions_delta_s), p=action_probs)
            except ValueError as e: best_action_idx = np.nanargmin(efes)
    selected_action_delta_s = candidate_actions_delta_s[best_action_idx]
    selected_action_efe = efes[best_action_idx] 
    
    return selected_action_delta_s, selected_action_efe, efes


# --- 模块自测试 (使用上一版本的测试用例，但会调用更新后的函数) ---
if __name__ == '__main__':
    # (与上一版本的测试用例相同，这里不再重复，以保持简洁)
    print("\n\n--- 测试完整AIF流程 (续，包含移植的custom_evidence和consensus) ---")
    num_agents_test_full = 2
    num_goals_test_full = 2
    # 使用原始论文的 's', 'A', 'B' 类型
    agent_types_test_full = ['s', 'A'] 
    
    agent_physical_params_list_full = [
        {"type": agent_types_test_full[0], "capabilities": ["distance_sensor"], 
         "evidence_sensitivity_eta": 30.0, "dist_evidence_weight":1.0, "agent_radius": 0.5,
         "vel_mag_thresh": 0.01, "dist_thresh_evidence": 200.0, "evidence_norm_factor": 1.0, "EPS":1e-9}, 
        {"type": agent_types_test_full[1], "capabilities": ["angle_sensor"], 
         "evidence_sensitivity_eta": 20.0, "angle_evidence_weight_factor": 1.0, "agent_radius": 0.5, # 注意：原始A类型只用角度
         "vel_mag_thresh": 0.01, "dist_thresh_evidence": 200.0, "evidence_norm_factor": 1.0, "EPS":1e-9}
    ]
    all_agent_ids_str_test_full = [f"Agent{i}" for i in range(num_agents_test_full)]
    goal_pos_test_full = [np.array([100.0, 0.0]), np.array([-100.0, 0.0])] 
    
    global_config_full = {
        "reasoning_level": 2, 
        "use_epistemic_planning": True, 
        "obs_variance_pos": 0.01, "obs_variance_vel": 0.001, # 减小观测噪声以突出证据差异
        "alpha_epistemic": 1.0, "beta_pragmatic": 0.8, "gamma_control_effort": 0.005,
        "policy_temperature": 0.001, 
        "belief_update_temperature": 1.0,
        "perspective_fusion_lambda": 0.5, # isobelief融合权重
    }
    
    _possible_G_temp_main = generate_possible_goal_configs(num_agents_test_full, num_goals_test_full, False)
    if _possible_G_temp_main.size > 0:
        _num_configs_temp_main = _possible_G_temp_main.shape[0]
        # 偏好 G = [Agent0->Goal0, Agent1->Goal1]
        # 假设 possible_G_configs 是 [[0,1], [1,0]] (如果num_agents=2, num_goals=2, no_repeat)
        # C_target for G=[0,1] is high, for G=[1,0] is low
        if _num_configs_temp_main == 2 : # 特别针对2x2不重复情况
             global_config_full["C_target_dist_G"] = np.array([0.9, 0.1]) 
        else:
             global_config_full["C_target_dist_G"] = np.ones(_num_configs_temp_main)/_num_configs_temp_main


    all_params_full = [initialize_agent_aif_params(
        all_agent_ids_str_test_full[i], i, all_agent_ids_str_test_full, num_goals_test_full,
        agent_physical_params_list_full[i], goal_pos_test_full, global_config_full,
        allow_multiple_to_same_goal=False
    ) for i in range(num_agents_test_full)]

    true_states_full = [ 
        np.array([95.0, 2.0, 1.0, 0.0]),    # Agent0 ('s') 接近 G0 (100,0), 速度指向G0
        np.array([-95.0, -2.0, -1.0, 0.0])  # Agent1 ('A') 接近 G1 (-100,0), 速度指向G1
    ]
    
    agent_to_decide_idx = 0
    agent_to_decide_params = all_params_full[agent_to_decide_idx]
    print(f"\n--- Agent {agent_to_decide_idx} (Type: {agent_to_decide_params['agent_type']}) 决策流程 (高阶) ---")
    if agent_to_decide_params["C_target_dist_G"] is not None and agent_to_decide_params["C_target_dist_G"].size > 0:
        print(f"  目标配置偏好 C_target_dist_G: {agent_to_decide_params['C_target_dist_G']}")
    else:
        print(f"  目标配置偏好 C_target_dist_G: 未定义或无效")

    agent0_observations = {}
    for target_idx_obs_loop in range(num_agents_test_full):
        agent0_observations[target_idx_obs_loop] = simulate_observation_for_agent(
            agent_to_decide_idx, target_idx_obs_loop, true_states_full, agent_to_decide_params
        )
    
    print(f"  Agent {agent_to_decide_idx} 的（模拟）观测:")
    for obs_target_idx_print, obs_val_vector_print in agent0_observations.items():
        print(f"    Obs of Agent {obs_target_idx_print} (type {all_params_full[obs_target_idx_print]['agent_type']}): pos=({obs_val_vector_print[0]:.1f},{obs_val_vector_print[1]:.1f}) vel=({obs_val_vector_print[2]:.1f},{obs_val_vector_print[3]:.1f})")

    log_L_G_agent0 = get_likelihood_for_agent(
        current_agent_idx=agent_to_decide_idx,
        current_agent_observations=agent0_observations,
        agent_params_list_all=all_params_full,
        true_states_for_simulation=true_states_full 
    )
    
    update_belief(agent_to_decide_idx, log_L_G_agent0, all_params_full) 
    # 打印时小心可能的空数组
    if agent_to_decide_params['possible_goal_configs_G'].size > 0 :
        print(f"  Agent {agent_to_decide_idx} 更新后的信念 Q(G) (configs: {agent_to_decide_params['possible_goal_configs_G']}): \n    {agent_to_decide_params['q_G_posterior']}")
    else:
        print(f"  Agent {agent_to_decide_idx} 更新后的信念 Q(G): 无有效配置")


    candidate_delta_s_options = [
        np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0]), 
        np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0, 0.0]), 
        np.array([0.0, 0.0, 0.0, 0.0]) 
    ]
    
    selected_ds, min_efe, all_efes = choice_heuristic(
        agent_idx=agent_to_decide_idx,
        candidate_actions_delta_s=candidate_delta_s_options,
        current_agent_physical_state=true_states_full[agent_to_decide_idx],
        all_agent_current_states_for_sim=true_states_full,
        agent_params_list_all=all_params_full
    )
    print(f"\n  Agent {agent_to_decide_idx} 候选动作的EFE值: {all_efes}")
    print(f"  Agent {agent_to_decide_idx} 选择的动作 (delta_s): {selected_ds[:2] if selected_ds is not None else 'None'}, 对应EFE: {min_efe:.3f}")