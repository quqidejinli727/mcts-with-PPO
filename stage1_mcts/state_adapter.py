"""
状态适配器
将NetAwareSearchState适配为我们的多网同构搜索需求
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from data_model import (
    NetAwareSearchState, FloorPlanRO, IsomorphicPinGroup, PinAssignment,
    SegmentUsage, Pin, Net, Segment
)

logger = logging.getLogger(__name__)

@dataclass
class PinStatus:
    """Pin状态标记"""
    pin_id: int
    is_allocated: bool = False
    is_currently_assigning: bool = False  # 正要分配的特殊标记
    assignment_order: int = 0  # 分配顺序，用于处理单元顺序
    
    def mark_as_assigning(self):
        """标记为正在分配"""
        self.is_currently_assigning = True
    
    def mark_as_allocated(self):
        """标记为已分配"""
        self.is_allocated = True
        self.is_currently_assigning = False

@dataclass
class MultiNetSearchAdapter:
    """多网搜索状态适配器"""
    
    base_state: NetAwareSearchState
    current_isomorphic_group: Optional[IsomorphicPinGroup] = None
    search_context: Any = None
    net_id_list: List[int] = None  # 添加网表ID列表，用于多网表处理
    current_net_index: int = 0     # 当前处理的网表索引
    pin_status_map: Dict[int, PinStatus] = None  # Pin状态映射
    
    def __post_init__(self):
        if self.search_context is None:
            self.search_context = SimpleSearchContext()
        if self.net_id_list is None:
            self.net_id_list = []
        if self.pin_status_map is None:
            self.pin_status_map = {}
        
        # 如果提供了同构组，初始化网表列表和pin状态
        if self.current_isomorphic_group:
            self._initialize_net_list()
            self._initialize_pin_status()
    
    def _initialize_segment_usage_manager(self):
        """初始化segment使用管理器 - 全局管理所有segment的使用状态"""
        if not self.base_state.segment_usages:
            self.base_state.segment_usages = {}
        
        # 确保所有segment都有使用记录
        for seg_id in range(self.floorplan.num_segments):
            if seg_id not in self.base_state.segment_usages:
                segment = self.floorplan.get_segment(seg_id)
                if segment:
                    self.base_state.segment_usages[seg_id] = SegmentUsage(
                        seg_id=seg_id,
                        used_capacity=0.0,
                        assigned_pins=[]
                    )
        
        logger.debug(f"初始化segment使用管理器: {len(self.base_state.segment_usages)}个segment")
    
    def _initialize_pin_status(self):
        """初始化pin状态映射"""
        if not self.current_isomorphic_group:
            return
        
        # 为同构组中的每个pin创建状态记录
        for i, pin_id in enumerate(self.current_isomorphic_group.pin_ids):
            self.pin_status_map[pin_id] = PinStatus(
                pin_id=pin_id,
                is_allocated=self.is_pin_allocated(pin_id),
                assignment_order=i  # 使用同构组中的顺序作为分配顺序
            )
        
        # logger.info(f"初始化pin状态映射: 同构组{self.current_isomorphic_group.group_id}包含{len(self.pin_status_map)}个pin")
    
    def _initialize_net_list(self):
        """从同构组初始化网表列表"""
        if not self.current_isomorphic_group:
            return
        
        net_ids = set()
        for pin_id in self.current_isomorphic_group.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin:
                net_ids.add(pin.net_id)
        
        self.net_id_list = sorted(list(net_ids))
    
    @property
    def floorplan(self) -> FloorPlanRO:
        """获取floorplan"""
        return self.base_state.floorplan
    
    @property
    def net_assignments(self) -> Dict[int, List[PinAssignment]]:
        """获取网表分配"""
        return self.base_state.net_assignments or {}
    
    @property
    def segment_usages(self) -> Dict[int, SegmentUsage]:
        """获取segment使用情况"""
        return self.base_state.segment_usages or {}
    
    def is_terminal(self) -> bool:
        """检查是否终止 - 适应新的处理流程"""
        # 如果没有同构组或网表列表，视为终止
        if not self.current_isomorphic_group or not self.net_id_list:
            raise(ValueError("没有可用的同构组或网表列表"))
        
        # 检查当前网表状态
        current_net = self.floorplan.get_net(self.base_state.current_net_id)
        if not current_net:
            raise(ValueError("找不到当前网表"))
        
        # 如果当前pin索引超出网表范围，表示处理完成
        if self.base_state.current_pin_idx >= len(current_net.pin_ids):
            return True
        
        # # 检查是否所有网表都已处理完成且当前同构组的所有pin都已分配
        # all_nets_processed = (self.current_net_index >= len(self.net_id_list) - 1 and
        #                      self.base_state.current_pin_idx >= len(current_net.pin_ids) - 1)
        
        # if all_nets_processed and self.current_isomorphic_group:
        #     # 检查当前同构组的所有pin是否都已分配
        #     for pin_id in self.current_isomorphic_group.pin_ids:
        #         if not self.is_pin_allocated(pin_id):
        #             return False  # 还有未分配的关键pin
        #     return True  # 所有关键pin都已分配
        
        return False
    
    def get_current_pin(self) -> Optional[int]:
        """获取当前要处理的pin - 增强调试信息"""
        if self.is_terminal():
            logger.debug(f"get_current_pin: 状态已终止, net_id={self.base_state.current_net_id}, pin_idx={self.base_state.current_pin_idx}")
            return None
        
        current_net = self.floorplan.get_net(self.base_state.current_net_id)
        if not current_net:
            logger.debug(f"get_current_pin: 找不到网表, net_id={self.base_state.current_net_id}")
            return None
        
        if self.base_state.current_pin_idx >= len(current_net.pin_ids):
            logger.debug(f"get_current_pin: pin索引超出范围, net_id={self.base_state.current_net_id}, "
                        f"pin_idx={self.base_state.current_pin_idx}, net_size={len(current_net.pin_ids)}")
            return None
        
        pin_id = current_net.pin_ids[self.base_state.current_pin_idx]
        logger.debug(f"get_current_pin: 返回当前pin, net_id={self.base_state.current_net_id}, "
                    f"pin_idx={self.base_state.current_pin_idx}, pin_id={pin_id}")
        return pin_id
    
    def get_legal_segments(self) -> List[int]:
        """获取合法segments"""
        # 获取当前pin的合法segments
        current_pin = self.get_current_pin()
        if not current_pin:
            return []
        
        pin_obj = self.floorplan.get_pin(current_pin)
        if not pin_obj:
            return []
        
        # 获取block的segments
        return self.floorplan.get_block_segments(pin_obj.block_id)
    
    def get_segment_usage(self, seg_id: int) -> SegmentUsage:
        """获取segment使用情况 - 全局管理"""
        # 确保segment使用管理器已初始化
        self._initialize_segment_usage_manager()
        
        # 返回指定segment的使用情况，如果不存在则创建默认的
        if seg_id not in self.base_state.segment_usages:
            segment = self.floorplan.get_segment(seg_id)
            if segment:
                self.base_state.segment_usages[seg_id] = SegmentUsage(
                    seg_id=seg_id,
                    used_capacity=0.0,
                    assigned_pins=[]
                )
            else:
                # 返回一个空的SegmentUsage作为fallback
                return SegmentUsage(seg_id=seg_id, used_capacity=0.0, assigned_pins=[])
        
        return self.base_state.segment_usages[seg_id]
    
    def get_all_segment_usages(self) -> Dict[int, SegmentUsage]:
        """获取所有segment的使用情况"""
        self._initialize_segment_usage_manager()
        return self.base_state.segment_usages.copy()
    
    def update_segment_usage(self, seg_id: int, new_usage: SegmentUsage) -> bool:
        """更新segment使用情况"""
        try:
            if seg_id not in self.base_state.segment_usages:
                logger.warning(f"尝试更新不存在的segment {seg_id} 使用情况")
                return False
            
            self.base_state.segment_usages[seg_id] = new_usage.copy()
            logger.debug(f"更新segment {seg_id} 使用情况: 容量={new_usage.used_capacity}, pins={len(new_usage.assigned_pins)}")
            return True
            
        except Exception as e:
            logger.error(f"更新segment {seg_id} 使用情况失败: {str(e)}")
            return False
    
    def get_segment_usage_summary(self) -> Dict[str, Any]:
        """获取segment使用情况摘要"""
        self._initialize_segment_usage_manager()
        
        total_segments = len(self.base_state.segment_usages)
        used_segments = sum(1 for usage in self.base_state.segment_usages.values() if usage.used_capacity > 0)
        total_capacity = sum(self.floorplan.get_segment(seg_id).max_capacity for seg_id in self.base_state.segment_usages.keys())
        total_used_capacity = sum(usage.used_capacity for usage in self.base_state.segment_usages.values())
        total_assigned_pins = sum(len(usage.assigned_pins) for usage in self.base_state.segment_usages.values())
        
        return {
            "total_segments": total_segments,
            "used_segments": used_segments,
            "unused_segments": total_segments - used_segments,
            "total_capacity": total_capacity,
            "total_used_capacity": total_used_capacity,
            "total_remaining_capacity": total_capacity - total_used_capacity,
            "capacity_utilization_rate": total_used_capacity / total_capacity if total_capacity > 0 else 0.0,
            "total_assigned_pins": total_assigned_pins,
            "segment_details": {
                seg_id: {
                    "used_capacity": usage.used_capacity,
                    "max_capacity": self.floorplan.get_segment(seg_id).max_capacity,
                    "assigned_pins_count": len(usage.assigned_pins),
                    "utilization_rate": usage.used_capacity / self.floorplan.get_segment(seg_id).max_capacity
                                       if self.floorplan.get_segment(seg_id).max_capacity > 0 else 0.0
                }
                for seg_id, usage in self.base_state.segment_usages.items()
            }
        }
    
    def check_segment_capacity_violations(self) -> List[Tuple[int, float]]:
        """检查segment容量违规情况"""
        violations = []
        
        self._initialize_segment_usage_manager()
        
        for seg_id, usage in self.base_state.segment_usages.items():
            segment = self.floorplan.get_segment(seg_id)
            if segment and usage.used_capacity > segment.max_capacity + 1e-6:  # 添加微小误差容忍
                violations.append((seg_id, usage.used_capacity - segment.max_capacity))
        
        return violations
    
    def reset_segment_usage(self, seg_id: int) -> bool:
        """重置指定segment的使用情况"""
        try:
            if seg_id in self.base_state.segment_usages:
                segment = self.floorplan.get_segment(seg_id)
                if segment:
                    self.base_state.segment_usages[seg_id] = SegmentUsage(
                        seg_id=seg_id,
                        used_capacity=0.0,
                        assigned_pins=[]
                    )
                    logger.debug(f"重置segment {seg_id} 使用情况")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"重置segment {seg_id} 使用情况失败: {str(e)}")
            return False
    
    def is_pin_allocated(self, pin_id: int) -> bool:
        """检查pin是否已分配 - 只查找当前搜索net的assignment"""
        current_net_id = self.base_state.current_net_id
        
        # 只检查当前net的分配
        if current_net_id in self.net_assignments:
            for assignment in self.net_assignments[current_net_id]:
                if (hasattr(assignment, 'pin_id') and
                    assignment.pin_id == pin_id and
                    hasattr(assignment, 'seg_id') and
                    assignment.seg_id != -1):
                    return True
        
        # for net_assignments in self.net_assignments.values():
        #     for assignment in net_assignments:
        #         if (hasattr(assignment, 'pin_id') and
        #             assignment.pin_id == pin_id and
        #             hasattr(assignment, 'seg_id') and
        #             assignment.seg_id != -1):
        #             return True
        return False
      
    def get_pin_assignment(self, pin_id: int) -> Optional[PinAssignment]:
        """获取pin的分配 - 只查找当前搜索net的assignment"""
        current_net_id = self.base_state.current_net_id
        
        # 只检查当前net的分配
        if current_net_id in self.net_assignments:
            for assignment in self.net_assignments[current_net_id]:
                if hasattr(assignment, 'pin_id') and assignment.pin_id == pin_id:
                    return assignment
        
        return None
    
    def get_pin_segment(self, pin_id: int) -> Optional[int]:
        """获取pin分配的segment"""
        assignment = self.get_pin_assignment(pin_id)
        return getattr(assignment, 'seg_id', None) if assignment else None
    
    def get_pin_coordinate(self, pin_id: int) -> Optional[Tuple[float, float]]:
        """获取pin的坐标"""
        assignment = self.get_pin_assignment(pin_id)
        return getattr(assignment, 'coord', None) if assignment else None
    
    def add_assignment(self, assignment: PinAssignment):
        """添加或更新分配 - 如果pin已存在则更新，否则添加"""
        net_id = self.base_state.current_net_id
        
        # 确保net_id存在
        if net_id not in self.net_assignments:
            self.base_state.net_assignments[net_id] = []
        
        # 检查是否已存在该pin的分配
        existing_assignment = None
        existing_index = -1
        
        for i, existing in enumerate(self.net_assignments[net_id]):
            if hasattr(existing, 'pin_id') and existing.pin_id == assignment.pin_id:
                existing_assignment = existing
                existing_index = i
                break
        
        if existing_assignment:
            # 更新现有分配
            logger.debug(f"更新现有分配: pin{assignment.pin_id} 从seg{existing_assignment.seg_id} 到seg{assignment.seg_id}")
            
            # 如果segment发生变化，需要更新segment使用情况
            if existing_assignment.seg_id != assignment.seg_id and existing_assignment.seg_id != -1 and assignment.seg_id != -1:
                # 减少旧segment的使用
                if existing_assignment.seg_id in self.segment_usages:
                    old_usage = self.segment_usages[existing_assignment.seg_id]
                    if assignment.pin_id in old_usage.assigned_pins:
                        old_usage.assigned_pins.remove(assignment.pin_id)
                        # 减少容量使用
                        pin = self.floorplan.get_pin(assignment.pin_id)
                        if pin and hasattr(pin, 'width'):
                            old_usage.used_capacity -= pin.width
                            logger.debug(f"减少旧segment{existing_assignment.seg_id}容量: {pin.width}")
            
            # 更新分配
            if assignment.seg_id != -1:
                self.base_state.net_assignments[net_id][existing_index] = assignment
                self._update_segment_usage_for_assignment(assignment)
            
        else:
            # 添加新分配
            logger.debug(f"添加新分配: pin{assignment.pin_id} -> seg{assignment.seg_id}")
            self.base_state.net_assignments[net_id].append(assignment)
            
            # 如果新分配是实际分配，更新segment使用情况
            if assignment.seg_id != -1:
                self._update_segment_usage_for_assignment(assignment)
    
    def _update_segment_usage_for_assignment(self, assignment: PinAssignment):
        """根据分配更新segment使用情况"""
        if assignment.seg_id == -1:
            return  # 跳过临时分配
        
        segment = self.floorplan.get_segment(assignment.seg_id)
        if not segment:
            logger.error(f"无法找到segment{assignment.seg_id}")
            return
        
        if assignment.seg_id in self.segment_usages:
            current_usage = self.segment_usages[assignment.seg_id]
            if hasattr(current_usage, 'assign_pin') and current_usage.assign_pin(assignment.pin_id, self.floorplan):
                logger.debug(f"更新segment{assignment.seg_id}: 添加pin{assignment.pin_id}")
            else:
                logger.warning(f"无法将pin{assignment.pin_id}分配到segment{assignment.seg_id}（容量不足）")
        else:
            # segment不存在于segment_usages中，创建新的SegmentUsage
            logger.debug(f"segment{assignment.seg_id}不存在于segment_usages中，创建新的SegmentUsage")
            from data_model import SegmentUsage
            new_usage = SegmentUsage(
                seg_id=assignment.seg_id,
                used_capacity=0.0,
                assigned_pins=[]
            )
            # 尝试分配pin
            if new_usage.assign_pin(assignment.pin_id, self.floorplan):
                self.segment_usages[assignment.seg_id] = new_usage
                logger.debug(f"创建并更新segment{assignment.seg_id}: 添加pin{assignment.pin_id}")
            else:
                logger.warning(f"无法将pin{assignment.pin_id}分配到新创建的segment{assignment.seg_id}")
    
    def advance_to_next_task(self):
        """推进到下一个任务 - 智能推进，支持阶段化处理"""
        
        current_net = self.floorplan.get_net(self.base_state.current_net_id)
        if not current_net:
            logger.debug(f"当前网表不存在: net_id={self.base_state.current_net_id}")
            return
        
        logger.debug(f"advance_to_next_task: 当前状态 - net_id={self.base_state.current_net_id}, "
                    f"pin_idx={self.base_state.current_pin_idx}, net_size={len(current_net.pin_ids)}, "
                    f"net_id_list={self.net_id_list}, current_net_index={self.current_net_index}")
        
        # 阶段1：处理普通pin（推进到下一个pin）
        if self.base_state.current_pin_idx < len(current_net.pin_ids) - 1:
            self.base_state.current_pin_idx += 1
            logger.debug(f"推进到下一个pin: net_id={self.base_state.current_net_id}, pin_idx={self.base_state.current_pin_idx}")
            return
        
        logger.debug(f"阶段1条件不满足: {self.base_state.current_pin_idx} < {len(current_net.pin_ids) - 1} = {self.base_state.current_pin_idx < len(current_net.pin_ids) - 1}")

        self.base_state.current_pin_idx = len(current_net.pin_ids)
    
    def set_key_isomorphic_pin(self, pin_id: int) -> bool:
        """设置当前要处理的关键同构pin"""
        if not self.current_isomorphic_group or pin_id not in self.current_isomorphic_group.pin_ids:
            return False
        
        pin = self.floorplan.get_pin(pin_id)
        if not pin:
            return False
        
        target_net = self.floorplan.get_net(pin.net_id)
        if not target_net or pin_id not in target_net.pin_ids:
            return False
        
        # 设置当前处理的关键同构pin
        self.base_state.current_net_id = pin.net_id
        self.base_state.current_pin_idx = target_net.pin_ids.index(pin_id)
        
        logger.debug(f"设置关键同构pin: pin_id={pin_id}, net_id={pin.net_id}, pin_idx={self.base_state.current_pin_idx}")
        return True
    
    def mark_pin_as_assigning(self, pin_id: int) -> bool:
        """标记pin为正在分配状态"""
        if pin_id not in self.pin_status_map:
            return False
        
        self.pin_status_map[pin_id].mark_as_assigning()
        logger.debug(f"标记pin{pin_id}为正在分配状态")
        return True
    
    def mark_pin_as_allocated(self, pin_id: int) -> bool:
        """标记pin为已分配状态"""
        if pin_id not in self.pin_status_map:
            return False
        
        self.pin_status_map[pin_id].mark_as_allocated()
        logger.debug(f"标记pin{pin_id}为已分配状态")
        return True
    
    def get_pin_status(self, pin_id: int) -> Optional[PinStatus]:
        """获取pin状态"""
        return self.pin_status_map.get(pin_id)
    
    def is_pin_currently_assigning(self, pin_id: int) -> bool:
        """检查pin是否正在分配"""
        status = self.get_pin_status(pin_id)
        return status.is_currently_assigning if status else False
    
    def get_unallocated_isomorphic_pins(self) -> List[int]:
        """获取未分配的同构pin列表"""
        if not self.current_isomorphic_group:
            return []
        
        unallocated_pins = []
        for pin_id in self.current_isomorphic_group.pin_ids:
            status = self.get_pin_status(pin_id)
            if status and not status.is_allocated:
                unallocated_pins.append(pin_id)
        
        return unallocated_pins
    
    def get_isomorphic_pins_by_processing_order(self) -> List[int]:
        """按照处理单元顺序获取同构pin列表"""
        if not self.current_isomorphic_group:
            return []
        
        # 获取所有同构pin及其分配顺序
        pin_order_pairs = []
        for pin_id in self.current_isomorphic_group.pin_ids:
            status = self.get_pin_status(pin_id)
            if status:
                pin_order_pairs.append((pin_id, status.assignment_order))
        
        # 按分配顺序排序
        pin_order_pairs.sort(key=lambda x: x[1])
        
        return [pin_id for pin_id, _ in pin_order_pairs]
    
    def get_currently_assigning_pins(self) -> List[int]:
        """获取当前正在分配的pin列表"""
        assigning_pins = []
        for pin_id, status in self.pin_status_map.items():
            if status.is_currently_assigning:
                assigning_pins.append(pin_id)
        
        return assigning_pins
    
    def add_assignment_with_status(self, assignment: PinAssignment):
        """添加分配并更新pin状态 - 包含segment使用管理"""
        # 添加分配
        self.add_assignment(assignment)
        
        # 更新segment使用情况（如果是实际分配）
        if hasattr(assignment, 'seg_id') and hasattr(assignment, 'pin_id'):
            seg_id = assignment.seg_id
            pin_id = assignment.pin_id
            
            # 跳过临时分配
            if seg_id != -1:
                current_usage = self.get_segment_usage(seg_id)
                if current_usage.assign_pin(pin_id, self.floorplan):
                    self.update_segment_usage(seg_id, current_usage)
                    logger.debug(f"分配pin{pin_id}到segment{seg_id}并更新使用状态")
                else:
                    logger.warning(f"分配pin{pin_id}到segment{seg_id}失败（容量不足）")
        
        # 更新pin状态
        if hasattr(assignment, 'pin_id'):
            self.mark_pin_as_allocated(assignment.pin_id)
    
    def update_pin_assignment(self, pin_id: int, new_seg_id: int, net_id: int) -> bool:
        """更新已存在pin的分配（用于从临时分配到实际分配的转换）"""
        try:
            # 查找现有的分配
            existing_assignment = None
            
            if net_id in self.net_assignments:
                for i, assignment in enumerate(self.net_assignments[net_id]):
                    if hasattr(assignment, 'pin_id') and assignment.pin_id == pin_id:
                        existing_assignment = assignment
                        break
            
            if existing_assignment is None:
                logger.warning(f"未找到pin{pin_id}在网表{net_id}中的现有分配")
                return False
            
            old_seg_id = existing_assignment.seg_id
            
            # 如果新分配和旧分配相同，无需更新
            if old_seg_id == new_seg_id:
                logger.debug(f"pin{pin_id}分配未变化: seg{old_seg_id}")
                return True
            
            # 更新分配
            existing_assignment.seg_id = new_seg_id
            
            # # 如果旧分配是实际分配（非临时），需要更新segment使用情况
            # if old_seg_id != -1:
            #     # 减少旧segment的使用
            #     old_usage = self.get_segment_usage(old_seg_id)
            #     if pin_id in old_usage.assigned_pins:
            #         old_usage.assigned_pins.remove(pin_id)
            #         # 减少容量使用（需要获取pin宽度）
            #         pin = self.floorplan.get_pin(pin_id)
            #         if pin and hasattr(pin, 'width'):
            #             old_usage.used_capacity -= pin.width
            #             logger.debug(f"更新旧segment{old_seg_id}: 移除pin{pin_id}, 容量减少{pin.width}")
            #     # 更新全局segment使用记录
            #     self.update_segment_usage(old_seg_id, old_usage)
            
            # 如果新分配是实际分配（非临时），更新segment使用情况
            if new_seg_id != -1:
                new_usage = self.get_segment_usage(new_seg_id)
                if new_usage.assign_pin(pin_id, self.floorplan):
                    logger.debug(f"更新新segment{new_seg_id}: 添加pin{pin_id}")
                    # 更新全局segment使用记录
                    self.update_segment_usage(new_seg_id, new_usage)
                else:
                    logger.warning(f"无法将pin{pin_id}分配到segment{new_seg_id}")
                    return False
            
            logger.info(f"更新pin{pin_id}分配: seg{old_seg_id} -> seg{new_seg_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新pin{pin_id}分配失败: {str(e)}", exc_info=True)
            return False
    
    def apply_net_assignments_with_updates(self, net_assignments: List[Tuple[int, int]], net_id: int) -> bool:
        """应用网表分配结果，支持更新现有分配"""
        try:
            success_count = 0
            
            for pin_id, seg_id in net_assignments:
                # 检查是否已存在分配
                existing_assignment = None
                if net_id in self.net_assignments:
                    for assignment in self.net_assignments[net_id]:
                        if hasattr(assignment, 'pin_id') and assignment.pin_id == pin_id:
                            existing_assignment = assignment
                            break
                
                if existing_assignment:
                    # 更新现有分配
                    if self.update_pin_assignment(pin_id, seg_id, net_id):
                        success_count += 1
                else:
                    # 创建新分配
                    assignment = PinAssignment(
                        pin_id=pin_id,
                        seg_id=seg_id,
                        net_id=net_id
                    )
                    self.add_assignment_with_status(assignment)
                    success_count += 1
            
            logger.info(f"网表{net_id}分配更新完成: 成功{success_count}/{len(net_assignments)}")
            return success_count == len(net_assignments)
            
        except Exception as e:
            logger.error(f"应用网表{net_id}分配更新失败: {str(e)}", exc_info=True)
            return False
    
    def apply_final_assignments_with_updates(self, final_assignments: Dict[int, List[Tuple[int, int]]]) -> bool:
        """应用最终的分配结果（所有网表），支持更新现有分配"""
        try:
            total_success = 0
            total_assignments = 0
            
            for net_id, net_assignments in final_assignments.items():
                total_assignments += len(net_assignments)
                
                if self.apply_net_assignments_with_updates(net_assignments, net_id):
                    total_success += len(net_assignments)
                else:
                    logger.warning(f"最终分配: 网表{net_id}部分分配更新失败")
            
            logger.info(f"最终分配更新完成: 成功{total_success}/{total_assignments}")
            return total_success == total_assignments
            
        except Exception as e:
            logger.error(f"应用最终分配更新失败: {str(e)}", exc_info=True)
            return False
    
    def get_current_assignment_state(self) -> Dict[str, Any]:
        """获取当前分配状态摘要"""
        return {
            "total_assigned_pins": sum(len(assignments) for assignments in self.net_assignments.values()),
            "total_nets_with_assignments": len([net_id for net_id, assignments in self.net_assignments.items() if assignments]),
            "segment_usage_summary": {
                seg_id: {
                    "used_capacity": usage.used_capacity,
                    "assigned_pins_count": len(usage.assigned_pins)
                }
                for seg_id, usage in self.segment_usages.items()
            },
            "pin_status_summary": {
                "total_pins": len(self.pin_status_map),
                "allocated_pins": sum(1 for status in self.pin_status_map.values() if status.is_allocated),
                "assigning_pins": sum(1 for status in self.pin_status_map.values() if status.is_currently_assigning),
                "unallocated_pins": sum(1 for status in self.pin_status_map.values() if not status.is_allocated)
            }
        }
    
    def clone(self) -> 'MultiNetSearchAdapter':
        """克隆状态 - 包含完整的网表信息，确保深拷贝隔离"""
        
        # 深拷贝segment使用情况，确保每个segment使用都是独立的副本
        new_segment_usages = {}
        if self.base_state.segment_usages:
            for seg_id, usage in self.base_state.segment_usages.items():
                if hasattr(usage, 'copy'):
                    new_segment_usages[seg_id] = usage.copy()
                else:
                    # 手动创建副本
                    new_segment_usages[seg_id] = SegmentUsage(
                        seg_id=usage.seg_id,
                        used_capacity=usage.used_capacity,
                        assigned_pins=usage.assigned_pins.copy() if hasattr(usage.assigned_pins, 'copy') else list(usage.assigned_pins)
                    )
        
        # 深拷贝网表分配
        new_net_assignments = {}
        if self.base_state.net_assignments:
            for net_id, assignments in self.base_state.net_assignments.items():
                new_assignments = []
                for assignment in assignments:
                    # 创建assignment的深拷贝
                    if hasattr(assignment, 'pin_id') and hasattr(assignment, 'seg_id') and hasattr(assignment, 'net_id'):
                        new_assignment = PinAssignment(
                            pin_id=assignment.pin_id,
                            seg_id=assignment.seg_id,
                            net_id=assignment.net_id
                        )
                        new_assignments.append(new_assignment)
                    else:
                        # 如果assignment结构不同，直接添加引用（fallback）
                        new_assignments.append(assignment)
                new_net_assignments[net_id] = new_assignments
        
        # 创建新的基础状态
        new_base_state = NetAwareSearchState(
            current_net_id=self.base_state.current_net_id,
            current_pin_idx=self.base_state.current_pin_idx,
            segment_usages=new_segment_usages,
            net_assignments=new_net_assignments,
            floorplan=self.base_state.floorplan,
            parent=None  # 新状态没有父节点
        )
        
        # 深拷贝pin状态映射
        new_pin_status_map = {}
        if self.pin_status_map:
            for pin_id, status in self.pin_status_map.items():
                new_pin_status_map[pin_id] = PinStatus(
                    pin_id=status.pin_id,
                    is_allocated=status.is_allocated,
                    is_currently_assigning=status.is_currently_assigning,
                    assignment_order=status.assignment_order
                )
        
        # 深拷贝其他列表和字典
        new_net_id_list = self.net_id_list.copy() if hasattr(self.net_id_list, 'copy') else list(self.net_id_list)
        
        return MultiNetSearchAdapter(
            base_state=new_base_state,
            current_isomorphic_group=self.current_isomorphic_group,
            search_context=self.search_context,
            net_id_list=new_net_id_list,
            current_net_index=self.current_net_index,
            pin_status_map=new_pin_status_map
        )
    
    def remove_pin_assignment(self, pin_id: int):
        """移除pin分配"""
        for net_id, assignments in self.net_assignments.items():
            self.base_state.net_assignments[net_id] = [
                assignment for assignment in assignments 
                if getattr(assignment, 'pin_id', None) != pin_id
            ]
    

@dataclass
class SimpleGroupState:
    """简化的组状态"""
    group_id: int
    is_assigned: bool = False
    assigned_segment: Optional[int] = None
    
    def assign(self, segment_id: int):
        """分配segment"""
        self.is_assigned = True
        self.assigned_segment = segment_id
    
    def is_fully_assigned(self) -> bool:
        """检查是否完全分配"""
        return self.is_assigned

@dataclass
class SimpleSearchContext:
    """简化的搜索上下文"""
    complexity_level: float = 0.5
    strategy: str = "BALANCED"
    relaxed_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.relaxed_constraints is None:
            self.relaxed_constraints = {}

def create_search_adapter(base_state: NetAwareSearchState, 
                         isomorphic_group: IsomorphicPinGroup) -> MultiNetSearchAdapter:
    """创建搜索适配器"""
    return MultiNetSearchAdapter(
        base_state=base_state,
        current_isomorphic_group=isomorphic_group
    )