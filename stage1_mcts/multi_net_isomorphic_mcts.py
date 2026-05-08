"""
多网同构引脚分配MCTS - 主入口
集成所有组件，提供统一的API
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from data_model import FloorPlanRO, IsomorphicPinGroup, NetAwareSearchState, SegmentUsage, PinAssignment
from state_adapter import MultiNetSearchAdapter
from simple_reward_function import SimpleRewardFunction, create_simple_reward_function
from simple_isomorphic_mcts import SimplifiedIsomorphicMCTS, create_simple_isomorphic_mcts

logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """MCTS配置参数"""
    num_simulations: int = 1000
    time_limit: float = 30.0
    exploration_constant: float = 1.414
    allocation_strategy: str = "collaborative"  # collaborative, sequential, adaptive, delayed
    enable_backtrack: bool = True
    backtrack_max_depth: int = 3

@dataclass
class AssignmentResult:
    """分配结果"""
    success: bool
    assignments: Dict[int, List[Tuple[int, int]]]  # net_id -> [(pin_id, seg_id), ...]
    total_wirelength: float
    constraint_satisfaction_rate: float
    search_statistics: Dict[str, Any]
    error_message: Optional[str] = None

class MultiNetIsomorphicMCTS:
    """多网同构引脚分配MCTS主类"""
    
    def __init__(self, floorplan: FloorPlanRO, config: MCTSConfig = None,
                 initial_segment_usages: Dict[int, SegmentUsage] = None,
                 initial_net_assignments: Dict[int, List[PinAssignment]] = None):
        self.floorplan = floorplan
        self.config = config or MCTSConfig()
        
        # 保存初始状态
        self.initial_segment_usages = initial_segment_usages or {}
        self.initial_net_assignments = initial_net_assignments or {}
        
        self.reward_function = create_simple_reward_function(floorplan=floorplan)
        
        self.mcts_searcher = create_simple_isomorphic_mcts(
            floorplan=floorplan,
            reward_function=self.reward_function,
            exploration_constant=self.config.exploration_constant
        )

        self.backtrack_mechanism = None
    
    def assign_isomorphic_group(self, isomorphic_group: IsomorphicPinGroup) -> AssignmentResult:
        """为同构组分配引脚 - 按net为单位进行MCTS搜索"""
        
        logger.info(f"开始为同构组{isomorphic_group.group_id}分配引脚，包含{len(isomorphic_group.pin_ids)}个pin")
        
        try:
            # 1. 获取同构组的所有net，按处理顺序排序
            net_ids = self._get_sorted_net_ids(isomorphic_group)
            logger.info(f"同构组网表列表: {len(net_ids)}个nets, 顺序: {net_ids[:5]}{'...' if len(net_ids) > 5 else ''}")

            if isomorphic_group.group_id == 27:
                a = 0
            
            # 2. 按net为单位进行MCTS搜索
            all_net_results = {}
            total_wirelength = 0.0
            total_constraint_satisfaction = 0.0
            successful_nets = 0
            
            for net_id in net_ids:
                logger.info(f"处理网表 {net_id}/{len(net_ids)}")
                
                # 为该net创建独立的MCTS搜索
                if net_id == 17:
                    logger.debug(f"调试信息: 同构组{isomorphic_group.group_id}处理网表{net_id}")
                    logger.debug(f"当前全局状态: segments={len(self.initial_segment_usages)}, nets={len(self.initial_net_assignments)}")
                    logger.debug(f"同构组信息: pins={len(isomorphic_group.pin_ids)}, net_id={net_id}")
                    
                net_result = self._assign_single_net(net_id, isomorphic_group)
                
                if net_result.success:
                    # 应用该net的分配结果到全局状态
                    net_assignments = net_result.assignments.get(net_id, [])
                    if self._apply_net_result_to_global_state(net_id, net_assignments, isomorphic_group):
                        all_net_results[net_id] = net_assignments
                        total_wirelength += net_result.total_wirelength
                        total_constraint_satisfaction += net_result.constraint_satisfaction_rate
                        successful_nets += 1
                        logger.info(f"网表 {net_id} 分配成功并应用到全局状态")
                    else:
                        logger.warning(f"网表 {net_id} 分配成功但应用到全局状态失败")
                        # 仍然记录为成功，但可能需要注意
                        all_net_results[net_id] = net_assignments
                        total_wirelength += net_result.total_wirelength
                        total_constraint_satisfaction += net_result.constraint_satisfaction_rate
                        successful_nets += 1
                else:
                    logger.warning(f"网表 {net_id} 分配失败: {net_result.error_message}")
                    # 继续处理其他网表，但记录失败
            
            # 3. 处理同构Pin的最终分配（关键步骤）
            final_assignments = self._finalize_isomorphic_assignments(all_net_results, isomorphic_group)
            
            # 4. 应用最终的同构Pin分配结果到全局状态
            if final_assignments:
                if self._apply_final_isomorphic_assignments_to_global_state(final_assignments, isomorphic_group):
                    logger.info(f"同构组{isomorphic_group.group_id}最终分配成功应用到全局状态")
                else:
                    logger.warning(f"同构组{isomorphic_group.group_id}最终分配应用到全局状态失败")
            
            # 5. 生成最终结果
            if successful_nets > 0:
                avg_constraint_satisfaction = total_constraint_satisfaction / successful_nets
                final_result = AssignmentResult(
                    success=True,
                    assignments=final_assignments,
                    total_wirelength=total_wirelength,
                    constraint_satisfaction_rate=avg_constraint_satisfaction,
                    search_statistics={"successful_nets": successful_nets, "total_nets": len(net_ids)},
                    error_message=None
                )
                logger.info(f"同构组{isomorphic_group.group_id}分配成功: 成功网表{successful_nets}/{len(net_ids)}, 线长={total_wirelength:.2f}")
                return final_result
            else:
                # 所有网表都失败
                error_msg = "所有网表分配失败"
                logger.error(error_msg)
                return AssignmentResult(
                    success=False,
                    assignments={},
                    total_wirelength=0.0,
                    constraint_satisfaction_rate=0.0,
                    search_statistics={"successful_nets": 0, "total_nets": len(net_ids)},
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"分配过程异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AssignmentResult(
                success=False,
                assignments={},
                total_wirelength=0.0,
                constraint_satisfaction_rate=0.0,
                search_statistics={},
                error_message=error_msg
            )
                
        except Exception as e:
            error_msg = f"分配过程异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AssignmentResult(
                success=False,
                assignments={},
                total_wirelength=0.0,
                constraint_satisfaction_rate=0.0,
                search_statistics={},
                error_message=error_msg
            )
    
    def _get_sorted_net_ids(self, isomorphic_group: IsomorphicPinGroup) -> List[int]:
        """获取同构组的所有net，按处理顺序排序"""
        
        net_ids = set()
        for pin_id in isomorphic_group.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin:
                net_ids.add(pin.net_id)
        
        # 转换为列表并排序，确保处理顺序的一致性
        sorted_net_ids = sorted(list(net_ids))
        
        logger.info(f"同构组{isomorphic_group.group_id}包含的网表: {len(sorted_net_ids)}个, 顺序: {sorted_net_ids[:10]}{'...' if len(sorted_net_ids) > 10 else ''}")
        
        return sorted_net_ids
    
    def _assign_single_net(self, net_id: int, isomorphic_group: IsomorphicPinGroup) -> AssignmentResult:
        """为单个net进行MCTS搜索 - 每个net对应一颗MCTS树，一个pin对应一个node"""
        
        logger.info(f"开始为网表{net_id}进行MCTS搜索")
        
        try:
            # 1. 获取该net的在同构组中的pin
            net_pins = []
            for pin_id in isomorphic_group.pin_ids:
                pin = self.floorplan.get_pin(pin_id)
                if pin and pin.net_id == net_id:
                    net_pins.append(pin_id)
            
            if not net_pins:
                logger.warning(f"网表{net_id}在同构组中没有找到对应的pin")
                return AssignmentResult(
                    success=False,
                    assignments={},
                    total_wirelength=0.0,
                    constraint_satisfaction_rate=0.0,
                    search_statistics={},
                    error_message=f"网表{net_id}在同构组中没有pin"
                )

            
            # 2. 创建该net的初始状态
            initial_state = self._create_net_initial_state(net_id, isomorphic_group)
            
            # 3. 执行MCTS搜索
            search_result = self.mcts_searcher.search(
                initial_state=initial_state,
                num_simulations=self.config.num_simulations // 10,  # 为每个net分配较少的模拟次数
                time_limit=self.config.time_limit / 5  # 分配较少的时间
            )
            
            # 4. 验证搜索结果
            # if search_result.best_state and not search_result.best_state.is_terminal():
            #     logger.warning(f"网表{net_id}的MCTS搜索未到达终止状态")
            #     return AssignmentResult(
            #         success=False,
            #         assignments={},
            #         total_wirelength=0.0,
            #         constraint_satisfaction_rate=0.0,
            #         search_statistics=search_result.search_statistics,
            #         error_message="MCTS搜索未完成"
            #     )
            
            # 5. 提取分配结果
            net_assignments = self._extract_net_assignments(search_result.best_state, net_id)
            
            #TODO:完善这一部分的不合法情况
            # 5.1 检验非复用pin的分配情况，为未分配的非复用pin选择剩余容量最大的segment
            # 传入state以获取当前全局状态的分配信息
            # net_assignments = self._validate_and_fix_non_reused_pins(
            #     net_id, search_result.best_state, isomorphic_group
            # )
            
            total_wirelength = 0.0
            
            # 6. 计算约束满足率
            constraint_satisfaction = self._calculate_net_constraint_satisfaction(search_result.best_state, net_id, isomorphic_group)
            
            logger.info(f"网表{net_id} MCTS搜索完成: 线长={total_wirelength:.2f}, 约束满足率={constraint_satisfaction:.2%}")
            
            return AssignmentResult(
                success=True,
                assignments={net_id: net_assignments},
                total_wirelength=total_wirelength,
                constraint_satisfaction_rate=constraint_satisfaction,
                search_statistics=search_result.search_statistics,
                error_message=None
            )
            
        except Exception as e:
            error_msg = f"网表{net_id} MCTS搜索异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AssignmentResult(
                success=False,
                assignments={},
                total_wirelength=0.0,
                constraint_satisfaction_rate=0.0,
                search_statistics={},
                error_message=error_msg
            )
    
    def _create_net_initial_state(self, net_id: int, isomorphic_group: IsomorphicPinGroup) -> MultiNetSearchAdapter:
        """为单个net创建初始搜索状态"""
        
        
        # 创建基础的NetAwareSearchState
        from data_model import NetAwareSearchState
        
        base_state = NetAwareSearchState(
            current_net_id=net_id,
            current_pin_idx=0,
            segment_usages=self.initial_segment_usages.copy(),
            net_assignments=self.initial_net_assignments.copy(),
            floorplan=self.floorplan
        )
        
        # 创建适配器 - 专注于当前net
        initial_state = MultiNetSearchAdapter(
            base_state=base_state,
            current_isomorphic_group=isomorphic_group,
            search_context=None,
            net_id_list=[net_id],  # 只包含当前net
            current_net_index=0
        )
        
        logger.debug(f"网表{net_id}初始状态创建完成")
        return initial_state
    
    def _extract_net_assignments(self, state: MultiNetSearchAdapter, net_id: int) -> List[Tuple[int, int]]:
        """从状态中提取指定net的分配结果"""
        
        assignments = []
        net_assignments = state.net_assignments.get(net_id, [])
        
        for assignment in net_assignments:
            if hasattr(assignment, 'pin_id') and hasattr(assignment, 'seg_id'):
                assignments.append((assignment.pin_id, assignment.seg_id))
        
        return assignments
    
    def _validate_and_fix_non_reused_pins(
        self,
        net_id: int,
        state: MultiNetSearchAdapter,
        isomorphic_group: IsomorphicPinGroup
    ) -> List[Tuple[int, int]]:
        """
        检验非复用pin的分配情况，为未分配的非复用pin选择剩余容量最大的segment
        
        逻辑：
        1. 从当前全局state中获取分配信息
        2. 检查当前net需要分配的所有非复用pin（单pin同构组的pin）
        3. 如果非复用pin的分配结果seg_id为-1（未分配），则为其选择可分配segment中剩余容量最大的segment
        4. 返回更新后的分配结果
        """
        # 从state中提取当前net的分配结果
        net_assignments_list = state.net_assignments.get(net_id, [])
        
        # 将分配结果转换为列表和字典方便查找
        net_assignments = []
        for assignment in net_assignments_list:
            if hasattr(assignment, 'pin_id') and hasattr(assignment, 'seg_id'):
                net_assignments.append((assignment.pin_id, assignment.seg_id))
        
        fixed_assignments = []
        fixed_count = 0
        
        # 从assignment结果中遍历每个pin
        for pin_id, current_seg_id in net_assignments:
            # 只处理seg_id为-1（未分配）的pin
            if current_seg_id != -1:
                # 已分配，保持原分配
                fixed_assignments.append((pin_id, current_seg_id))
                continue
            
            # 检查该pin是否是复用pin（属于多pin同构组）
            pin = self.floorplan.get_pin(pin_id)
            if not pin:
                fixed_assignments.append((pin_id, -1))
                continue
            
            pin_iso_group = self.floorplan.get_pin_isomorphic_group(pin_id)
            # 如果该pin属于多pin的同构组（大小>1），则是复用pin，跳过处理
            if pin_iso_group and pin_iso_group.size > 1:
                fixed_assignments.append((pin_id, -1))
                continue
            
            # 非复用pin（单pin同构组），需要选择剩余容量最大的segment
                # 未分配的非复用pin，需要选择剩余容量最大的segment
            candidate_segments = self.floorplan.get_pin_segments(pin_id)
                
            if not candidate_segments:
                logger.warning(f"Pin {pin_id} 没有可分配的segment")
                fixed_assignments.append((pin_id, -1))
                continue
                
            # 选择剩余容量最大的segment
                max_remaining_capacity = -1.0
                best_seg_id = -1
                
                for seg_id in candidate_segments:
                    segment = self.floorplan.get_segment(seg_id)
                    if segment and hasattr(segment, 'max_capacity'):
                        # 从当前state获取segment使用量
                        current_usage = state.get_segment_usage(seg_id)
                        
                        remaining_capacity = segment.max_capacity - current_usage
                        if remaining_capacity > max_remaining_capacity:
                            max_remaining_capacity = remaining_capacity
                            best_seg_id = seg_id
                
                if best_seg_id != -1:
                    fixed_assignments.append((pin_id, best_seg_id))
                    fixed_count += 1
                    logger.info(f"为未分配的非复用pin {pin_id} 选择剩余容量最大的segment {best_seg_id} "
                               f"(剩余容量: {max_remaining_capacity:.2f})")
                else:
                    fixed_assignments.append((pin_id, -1))
            else:
                # 已分配，保持原分配
                fixed_assignments.append((pin_id, current_seg_id))
        
        if fixed_count > 0:
            logger.info(f"网表{net_id}: 为{fixed_count}个未分配的非复用pin修复了分配")
        
        return fixed_assignments
    
    def _calculate_net_constraint_satisfaction(self, state: MultiNetSearchAdapter, net_id: int, isomorphic_group: IsomorphicPinGroup) -> float:
        """计算指定net的约束满足率"""
        
        # 检查该net的所有pin是否都已分配
        net_pins = []
        for pin_id in isomorphic_group.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin and pin.net_id == net_id:
                net_pins.append(pin_id)
        
        if not net_pins:
            return 0.0
        
        # 检查容量约束
        capacity_violations = self._check_capacity_constraints(state)
        capacity_penalty = min(0.2 * capacity_violations, 0.5)  # 最多扣除50%
        
        return max(0.0, 1.0 - capacity_penalty)
    
    def _create_initial_state(self, isomorphic_group: IsomorphicPinGroup) -> MultiNetSearchAdapter:
        """创建初始搜索状态 - 使用当前全局状态"""
        
        logger.info(f"创建同构组{isomorphic_group.group_id}的初始搜索状态")
        logger.info(f"使用全局状态: {len(self.initial_segment_usages)} segments, "
                   f"{len(self.initial_net_assignments)} nets 已分配")
        
        # 从pin_ids中提取对应的net_ids
        net_ids = set()
        for pin_id in isomorphic_group.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin:
                net_ids.add(pin.net_id)
        
        net_ids = sorted(list(net_ids))
        logger.info(f"同构组信息: pins={len(isomorphic_group.pin_ids)}, nets={len(net_ids)}, net_ids={net_ids[:5]}{'...' if len(net_ids) > 5 else ''}")
        
        # 确定起始网表 - 使用同构组中的第一个网表
        start_net_id = net_ids[0] if net_ids else 0
        logger.info(f"选择起始网表: {start_net_id}")
        
        # 创建基础的NetAwareSearchState - 使用全局状态
        from data_model import NetAwareSearchState
        
        base_state = NetAwareSearchState(
            current_net_id=start_net_id,  # 从同构组的第一个net开始
            current_pin_idx=0,
            segment_usages=self.initial_segment_usages.copy(),  # 使用全局segment状态
            net_assignments=self.initial_net_assignments.copy(),  # 使用全局net分配状态
            floorplan=self.floorplan
        )
        
        # 创建适配器 - 传入网表列表用于多网表处理
        initial_state = MultiNetSearchAdapter(
            base_state=base_state,
            current_isomorphic_group=isomorphic_group,
            processing_queue=[],
            group_states={},
            search_context=None,
            net_id_list=net_ids,  # 传入网表ID列表
            current_net_index=0   # 从第一个网表开始
        )
        
        # 验证初始状态
        logger.info(f"初始状态创建完成: net_id={initial_state.base_state.current_net_id}, pin_idx={initial_state.base_state.current_pin_idx}")
        logger.info(f"初始状态检查: is_terminal={initial_state.is_terminal()}, current_pin={initial_state.get_current_pin()}")
        logger.info(f"全局segment使用情况: {len(initial_state.segment_usages)} segments")
        
        return initial_state
    
    def _check_capacity_constraints(self, state: MultiNetSearchAdapter) -> int:
        """检查容量约束违反数量"""
        
        violations = 0
        
        for seg_id, usage in state.segment_usages.items():
            segment = self.floorplan.get_segment(seg_id)
            if segment and usage.used_capacity > segment.max_capacity:
                violations += 1
        
        return violations
    
    def _calculate_total_wirelength(self, state: MultiNetSearchAdapter) -> float:
        """计算总线长"""
        
        total_wirelength = 0.0
        
        for net_id, assignments in state.net_assignments.items():
            if len(assignments) >= 2:
                net_wirelength = self._calculate_net_wirelength(assignments)
                total_wirelength += net_wirelength
        
        return total_wirelength
    
    def _calculate_net_wirelength(self, assignments: List) -> float:
        """计算单个net的线长"""
        
        # 使用HPWL计算
        if len(assignments) < 2:
            return 0.0
        
        x_coords = []
        y_coords = []
        
        for assignment in assignments:
            if hasattr(assignment, 'coord') and assignment.coord:
                x_coords.append(assignment.coord[0])
                y_coords.append(assignment.coord[1])
        
        if not x_coords or not y_coords:
            return 0.0
        
        hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        return hpwl
    
    def _generate_assignment_result(self, search_result, validation_result) -> AssignmentResult:
        """生成分配结果"""
        
        assignments = self._extract_assignments(search_result.best_state)
        
        return AssignmentResult(
            success=True,
            assignments=assignments,
            total_wirelength=validation_result['total_wirelength'],
            constraint_satisfaction_rate=validation_result['constraint_satisfaction_rate'],
            search_statistics=search_result.search_statistics
        )
    
    #TODO:对不存在合法segment的修改
    def _get_legal_segments_for_isomorphic_group(self, isomorphic_group: IsomorphicPinGroup, current_pin_capacity: float) -> List[int]:
        """获取同构组所有pin的合法segment位置 - 简化版，直接从floorplan获取"""
        
        legal_segments = set()
        all_candidate_segments = set()  # 收集所有候选segment
        
        # 获取同构组中所有pin的合法segment
        for pin_id in isomorphic_group.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if not pin:
                continue
            
            # 直接从floorplan获取该pin可能分配的segment
            candidate_segments = self.floorplan.get_pin_segments(pin_id)
            all_candidate_segments.update(candidate_segments)
            
            # 过滤掉容量已满的segment
            for seg_id in candidate_segments:
                segment = self.floorplan.get_segment(seg_id)
                if segment and hasattr(segment, 'max_capacity'):
                    current_usage = 0.0
                    if seg_id in self.initial_segment_usages:
                        usage = self.initial_segment_usages[seg_id]
                        if hasattr(usage, 'used_capacity'):
                            current_usage = usage.used_capacity
                        else:
                            logger.warning(f"segment{seg_id}的usage对象没有used_capacity属性")
                    else:
                        logger.debug(f"segment{seg_id}不存在于initial_segment_usages中")
                    if current_usage + current_pin_capacity < segment.max_capacity - 1e-6:  # 增加微小误差限制
                        legal_segments.add(seg_id)

        # 如果没有合法segment，选择容量剩余最多的segment
        if not legal_segments and all_candidate_segments:
            max_remaining_capacity = -1.0
            best_segment_id = None
            
            for seg_id in all_candidate_segments:
                segment = self.floorplan.get_segment(seg_id)
                if segment and hasattr(segment, 'max_capacity'):
                    current_usage = 0.0
                    if seg_id in self.initial_segment_usages:
                        usage = self.initial_segment_usages[seg_id]
                        if hasattr(usage, 'used_capacity'):
                            current_usage = usage.used_capacity
                    remaining_capacity = segment.max_capacity - current_usage
                    if remaining_capacity > max_remaining_capacity:
                        max_remaining_capacity = remaining_capacity
                        best_segment_id = seg_id
            
            if best_segment_id is not None:
                legal_segments.add(best_segment_id)
                logger.warning(f"没有合法segment，选择容量剩余最多的segment {best_segment_id} "
                              f"(剩余容量: {max_remaining_capacity:.2f})")
        
        return list(legal_segments)
    
    def _finalize_isomorphic_assignments(self, all_net_results: Dict[int, List[Tuple[int, int]]],
                                       isomorphic_group: IsomorphicPinGroup) -> Dict[int, List[Tuple[int, int]]]:
        """处理同构Pin的最终分配 - 正确的逻辑：先选定合法位置，然后更新所有nets中的同构Pin分配"""
        
        logger.info(f"开始处理同构组{isomorphic_group.group_id}的最终分配，涉及{len(all_net_results)}个网表")
        
        try:
            # 1. 获取同构组所有pin的合法segment位置
            current_pin_capcity = self.floorplan.get_pin(isomorphic_group.pin_ids[0]).width if isomorphic_group.pin_ids else 0.0
            legal_segments = self._get_legal_segments_for_isomorphic_group(isomorphic_group, current_pin_capcity)
            logger.info(f"同构组{isomorphic_group.group_id}有{len(legal_segments)}个合法segment位置")
            
            if not legal_segments:
                logger.warning("没有找到合法的segment位置，返回原始分配结果")
                return all_net_results
            
            # 2. 遍历所有可能的segment位置，为每个位置评估整体分配方案
            best_overall_assignments = {}
            best_overall_reward = float('-inf')
            best_selected_segment = None
            
            for candidate_seg_id in legal_segments:
                logger.debug(f"评估选定segment {candidate_seg_id} 的整体分配方案")
                
                # 为每个net创建更新后的分配方案
                current_assignments = {}
                total_current_reward = 0.0
                
                for net_id, original_assignments in all_net_results.items():
                    # 获取该net的同构pin
                    net_isomorphic_pins = []
                    for pin_id in isomorphic_group.pin_ids:
                        pin = self.floorplan.get_pin(pin_id)
                        if pin and pin.net_id == net_id:
                            net_isomorphic_pins.append(pin_id)
                    
                    if not net_isomorphic_pins:
                        # 该net没有同构pin，保持原始分配
                        current_assignments[net_id] = original_assignments
                        continue
                    
                    # 创建更新后的分配：替换同构Pin的临时分配（seg_id=-1）为选定的segment
                    updated_assignments = []
                    
                    for pin_id, seg_id in original_assignments:
                        if pin_id in net_isomorphic_pins and seg_id == -1:
                            # 这是同构Pin的临时分配，替换为选定的segment
                            updated_assignments.append((pin_id, candidate_seg_id))
                            logger.debug(f"网表{net_id}: 更新同构Pin{pin_id}分配: -1 -> {candidate_seg_id}")
                        else:
                            # 保持原有分配
                            updated_assignments.append((pin_id, seg_id))
                    
                    # 计算这个net更新后的reward - 转换updated_assignments为PinAssignment列表
                    pin_assignments = []
                    for pin_id, seg_id in updated_assignments:
                        assignment = PinAssignment(
                            pin_id=pin_id,
                            seg_id=seg_id,
                            net_id=net_id
                        )
                        pin_assignments.append(assignment)
                    
                    net_reward = self.reward_function.compute_reward(pin_assignments)
                    
                    current_assignments[net_id] = updated_assignments
                    total_current_reward += net_reward
                    
                    logger.debug(f"网表{net_id}, segment{candidate_seg_id}: reward={net_reward:.4f}")
                
                # 计算整体reward（可以添加跨net的一致性奖励等）
                # 这里简单使用各net reward的总和
                overall_reward = total_current_reward
                
                logger.debug(f"选定segment {candidate_seg_id}: 整体reward={overall_reward:.4f}")
                
                # 选择整体reward最好的方案
                if overall_reward > best_overall_reward:
                    best_overall_reward = overall_reward
                    best_overall_assignments = current_assignments
                    best_selected_segment = candidate_seg_id
            
            if best_selected_segment is not None:
                logger.info(f"最佳整体方案: 选定segment {best_selected_segment}, 整体reward={best_overall_reward:.4f}")
                final_assignments = best_overall_assignments
            else:
                logger.warning("未找到合适的分配方案，返回原始分配")
                final_assignments = all_net_results
            
            # 3. 验证最终分配的同构一致性
            consistency_rate = self._validate_final_consistency(final_assignments, isomorphic_group)
            logger.info(f"最终分配一致性检查: {consistency_rate:.2%}")
            
            if consistency_rate < 0.8:  # 一致性阈值
                logger.warning(f"最终分配一致性不足: {consistency_rate:.2%}")
                # 尝试调整分配以提高一致性
            
            logger.info(f"同构组{isomorphic_group.group_id}最终分配完成: 涉及{len(final_assignments)}个网表")
            return final_assignments
            
        except Exception as e:
            logger.error(f"最终分配处理异常: {str(e)}", exc_info=True)
            # 返回原始的net结果作为fallback
            return all_net_results
    
    def _create_temp_state_for_reward(self, net_id: int, net_assignments: List[Tuple[int, int]],
                                    isomorphic_group: IsomorphicPinGroup) -> MultiNetSearchAdapter:
        """创建用于reward计算的临时状态"""
        
        from data_model import NetAwareSearchState
        
        # 创建基础状态
        base_state = NetAwareSearchState(
            current_net_id=net_id,
            current_pin_idx=0,
            segment_usages=self.initial_segment_usages.copy(),
            net_assignments={net_id: []},  # 只包含当前net
            floorplan=self.floorplan
        )
        
        # 先创建适配器，获取完整的temp_state
        temp_state = MultiNetSearchAdapter(
            base_state=base_state,
            current_isomorphic_group=isomorphic_group,
            search_context=None,
            net_id_list=[net_id],
            current_net_index=0
        )
        
        # 添加分配结果到temp_state - 直接实现，不依赖外部方法
        for pin_id, seg_id in net_assignments:
            # 创建分配对象
            assignment = PinAssignment(
                pin_id=pin_id,
                seg_id=seg_id,
                net_id=net_id
            )
            
            # 添加到temp_state的net_assignments（通过属性访问器）
            current_assignments = temp_state.net_assignments
            if net_id not in current_assignments:
                current_assignments[net_id] = []
            current_assignments[net_id].append(assignment)
        
        return temp_state
    
    def _validate_final_consistency(self, final_assignments: Dict[int, List[Tuple[int, int]]],
                                  isomorphic_group: IsomorphicPinGroup) -> float:
        """验证最终分配的同构一致性"""
        
        if not final_assignments or not isomorphic_group.pin_ids:
            return 0.0
        
        # 检查同构pin是否被分配到相似的segment
        consistent_assignments = 0
        total_assignments = 0
        
        # 获取同构pin的分配模式
        pin_assignments = {}
        for net_id, assignments in final_assignments.items():
            for pin_id, seg_id in assignments:
                if pin_id in isomorphic_group.pin_ids:
                    if pin_id not in pin_assignments:
                        pin_assignments[pin_id] = []
                    pin_assignments[pin_id].append(seg_id)
        
        # 检查分配一致性
        for pin_id, seg_ids in pin_assignments.items():
            if len(seg_ids) > 0:
                total_assignments += 1
                # 检查这些segment是否具有相同的master和索引
                if self._check_segments_consistency(seg_ids):
                    consistent_assignments += 1
        
        return consistent_assignments / max(1, total_assignments)
    
    def _check_segments_consistency(self, seg_ids: List[int]) -> bool:
        """检查多个segment是否具有一致性（相同的master和索引）"""
        
        if not seg_ids:
            return True
        
        if len(seg_ids) == 1:
            return True
        
        # 获取第一个segment作为参考
        ref_seg = self.floorplan.get_segment(seg_ids[0])
        if not ref_seg:
            return False
        
        ref_master_id = getattr(ref_seg, 'master_id', None)
        ref_index = self.floorplan.get_segment_index_in_master(seg_ids[0], ref_master_id)
        
        # 检查其他segment是否匹配
        for seg_id in seg_ids[1:]:
            seg = self.floorplan.get_segment(seg_id)
            if not seg:
                return False
            
            master_id = getattr(seg, 'master_id', None)
            seg_index = self.floorplan.get_segment_index_in_master(seg_id, master_id)
            
            if master_id != ref_master_id or seg_index != ref_index:
                return False
        
        return True
    
    
    def _apply_net_result_to_global_state(self, net_id: int, net_assignments: List[Tuple[int, int]],
                                        isomorphic_group: IsomorphicPinGroup) -> bool:
        """将单个net的分配结果应用到全局状态"""
        try:
            logger.info(f"应用网表{net_id}分配结果到全局状态: {len(net_assignments)}个分配")
            
            # 创建临时状态来应用分配
            temp_state = MultiNetSearchAdapter(
                base_state=NetAwareSearchState(
                    current_net_id=net_id,
                    current_pin_idx=0,
                    segment_usages=self.initial_segment_usages.copy(),
                    net_assignments=self.initial_net_assignments.copy(),
                    floorplan=self.floorplan
                ),
                current_isomorphic_group=isomorphic_group,
                search_context=None,
                net_id_list=[net_id],
                current_net_index=0
            )
            
            # 应用分配结果
            success = temp_state.apply_net_assignments_with_updates(net_assignments, net_id)
            
            if success:
                # 更新全局状态
                self.initial_segment_usages = temp_state.segment_usages.copy()
                self.initial_net_assignments = temp_state.net_assignments.copy()
                logger.info(f"网表{net_id}分配结果成功应用到全局状态")
                
                # 记录状态摘要
                state_summary = temp_state.get_current_assignment_state()
                logger.debug(f"网表{net_id}应用后状态: {state_summary}")
            else:
                logger.warning(f"网表{net_id}分配结果应用到全局状态失败")
            
            return success
            
        except Exception as e:
            logger.error(f"应用网表{net_id}分配结果到全局状态失败: {str(e)}", exc_info=True)
            return False
    
    def _apply_final_isomorphic_assignments_to_global_state(self, final_assignments: Dict[int, List[Tuple[int, int]]],
                                                           isomorphic_group: IsomorphicPinGroup) -> bool:
        """将最终的同构Pin分配结果应用到全局状态"""
        try:
            total_assignments = sum(len(assignments) for assignments in final_assignments.values())
            logger.info(f"应用最终同构组分配结果到全局状态: {total_assignments}个分配，涉及{len(final_assignments)}个网表")
            
            # 创建临时状态来应用最终分配
            temp_state = MultiNetSearchAdapter(
                base_state=NetAwareSearchState(
                    current_net_id=0,
                    current_pin_idx=0,
                    segment_usages=self.initial_segment_usages.copy(),
                    net_assignments=self.initial_net_assignments.copy(),
                    floorplan=self.floorplan
                ),
                current_isomorphic_group=isomorphic_group,
                search_context=None,
                net_id_list=list(final_assignments.keys()),
                current_net_index=0
            )
            
            # 应用最终分配结果
            success = temp_state.apply_final_assignments_with_updates(final_assignments)
            
            if success:
                # 更新全局状态
                self.initial_segment_usages = temp_state.segment_usages.copy()
                self.initial_net_assignments = temp_state.net_assignments.copy()
                logger.info(f"最终同构组分配结果成功应用到全局状态")
                
                # 记录最终状态摘要
                state_summary = temp_state.get_current_assignment_state()
                # logger.info(f"最终分配状态摘要: {state_summary}")
            else:
                logger.warning(f"最终同构组分配结果应用到全局状态失败")
            
            return success
            
        except Exception as e:
            logger.error(f"应用最终同构组分配结果到全局状态失败: {str(e)}", exc_info=True)
            return False
    
    def _extract_assignments(self, state: MultiNetSearchAdapter) -> Dict[int, List[Tuple[int, int]]]:
        """提取分配结果"""
        
        assignments = {}
        
        for net_id, net_assignments in state.net_assignments.items():
            net_result = []
            for assignment in net_assignments:
                if hasattr(assignment, 'pin_id') and hasattr(assignment, 'seg_id'):
                    net_result.append((assignment.pin_id, assignment.seg_id))
            assignments[net_id] = net_result
        
        return assignments
    
    def get_analytics(self) -> Dict[str, Any]:
        """获取算法分析数据"""
        
        analytics = {
            "constraint_manager": self.constraint_manager.get_constraint_analytics(),
            "reward_function": self.reward_function.get_reward_analytics(),
            "mcts_searcher": {
                "total_simulations": self.mcts_searcher.total_simulations,
                "best_reward_found": self.mcts_searcher.best_reward_found
            }
        }
        
        if self.backtrack_mechanism:
            analytics["backtrack_mechanism"] = self.backtrack_mechanism.get_backtrack_analytics()
        
        return analytics
    
    def reset_statistics(self):
        """重置统计信息"""
        self.constraint_manager.reset_statistics()
        self.reward_function.clear_cache()
        
        if self.backtrack_mechanism:
            self.backtrack_mechanism.reset_statistics()

# 工厂函数
def create_multi_net_isomorphic_mcts(floorplan: FloorPlanRO, 
                                   config: MCTSConfig = None) -> MultiNetIsomorphicMCTS:
    """创建多网同构MCTS分配器"""
    return MultiNetIsomorphicMCTS(floorplan=floorplan, config=config)

# 便捷函数
def assign_isomorphic_pins(floorplan: FloorPlanRO, 
                         isomorphic_group: IsomorphicPinGroup,
                         num_simulations: int = 1000,
                         time_limit: float = 30.0,
                         allocation_strategy: str = "adaptive") -> AssignmentResult:
    """便捷函数：直接为同构组分配引脚"""
    
    config = MCTSConfig(
        num_simulations=num_simulations,
        time_limit=time_limit,
        allocation_strategy=allocation_strategy
    )
    
    mcts_allocator = create_multi_net_isomorphic_mcts(floorplan, config)
    return mcts_allocator.assign_isomorphic_group(isomorphic_group)