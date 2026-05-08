"""
最终的精细拓扑排序算法
实现方案：连通分量 → 同构组优先级排序 → 核心Pin优先 → 相关nets顺序
加入连通分量net数排序作为第一优先级
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from data_model import FloorPlanRO

logger = logging.getLogger(__name__)

@dataclass
class IsomorphicGroupInfo:
    """同构组信息"""
    group_id: int
    pin_ids: List[int]           # 组内所有Pin ID
    involved_nets: Set[int]      # 涉及的nets
    priority_score: float        # 优先级分数
    core_pin_id: Optional[int]   # 核心Pin ID（优先级最高的Pin）
    processing_order: int = 0    # 处理顺序

@dataclass
class ProcessingUnit:
    """处理单元"""
    component_id: int            # 连通分量ID
    component_size: int          # 连通分量大小（net数）
    component_sort_order: int    # 连通分量排序（按大小）
    isomorphic_group_id: int     # 同构组ID
    net_ids: List[int]           # 涉及的nets（按优先级排序）
    is_reuse_critical: bool      # 是否复用关键
    core_pin_id: Optional[int]   # 核心Pin ID
    processing_stage: int        # 处理阶段
    priority_score: float        # 优先级分数

class FinalTopologyBuilder:
    """最终拓扑构建器"""
    
    def __init__(self, floorplan: FloorPlanRO):
        self.floorplan = floorplan
        self.net_scores = {}  # net_id -> score
        self.isomorphic_groups = {}  # group_id -> IsomorphicGroupInfo
        
    def build_final_topology(self) -> List[ProcessingUnit]:
        """
        构建最终的精细拓扑排序
        
        核心算法：
        1. 连通分量分解，按net数从多到少排序
        2. 每个连通分量内：同构组优先级排序
        3. 每个同构组内：核心Pin优先，然后相关nets
        4. 标注核心Pin和合并层级
        
        Returns:
            处理单元列表，按精细顺序排列
        """
        logger.info("开始构建最终精细拓扑排序...")
        
        # 步骤1：计算net和同构组打分
        logger.info("步骤1：计算打分信息...")
        self._calculate_scores()
        
        # 步骤2：构建无向图并分解连通分量
        logger.info("步骤2：连通分量分解...")
        components = self._find_connected_components()
        
        # 步骤3：构建同构组信息
        logger.info("步骤3：构建同构组信息...")
        self._build_isomorphic_groups()
        
        # 步骤4：确定同构组处理顺序
        logger.info("步骤4：确定同构组处理顺序...")
        self._determine_isomorphic_group_order()
        
        # 步骤5：生成精细处理顺序（按连通分量net数排序）
        logger.info("步骤5：生成精细处理顺序（按连通分量net数排序）...")
        processing_units = self._generate_processing_order(components)
        
        logger.info(f"最终拓扑排序构建完成：{len(processing_units)}个处理单元")
        self._print_refined_stats(processing_units)
        
        return processing_units
    
    def _calculate_scores(self):
        """计算net和同构组打分"""
        logger.info("计算打分信息...")
        
        # 计算每个net的分数
        for net_id in range(self.floorplan.num_nets):
            net = self.floorplan.get_net(net_id)
            
            # 模块层次
            hierarchy = self._calculate_module_hierarchy(net_id)
            
            # 复杂度
            complexity = self._calculate_complexity_score(net_id)
            
            # 综合分数
            total_score = (100 - hierarchy) * 10 + (1.0 - complexity) * 5
            
            self.net_scores[net_id] = total_score
        
        logger.info(f"net打分计算完成：{len(self.net_scores)}个nets")
    
    def _calculate_module_hierarchy(self, net_id: int) -> int:
        """计算net的模块层次"""
        net = self.floorplan.get_net(net_id)
        hierarchy_levels = set()
        
        for pin_id in net.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin:
                block = self.floorplan.get_block(pin.block_id)
                level = block.name.count('.')  # 用点号数量作为层次指标
                hierarchy_levels.add(level)
        
        return int(sum(hierarchy_levels) / len(hierarchy_levels)) if hierarchy_levels else 0
    
    def _calculate_complexity_score(self, net_id: int) -> float:
        """计算net的处理复杂度分数"""
        net = self.floorplan.get_net(net_id)
        
        # 1. Pin数量因子（归一化到0-0.4）
        pin_count = len(net.pin_ids)
        max_pins = max(len(self.floorplan.get_net(i).pin_ids) for i in range(self.floorplan.num_nets))
        pin_factor = min(0.4, (pin_count / max_pins) * 0.4) if max_pins > 0 else 0
        
        # 2. 同构组因子（0-0.3）
        reuse_groups = set()
        for pin_id in net.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin and pin.isomorphic_group_id is not None:
                reuse_groups.add(pin.isomorphic_group_id)
        reuse_factor = min(0.3, len(reuse_groups) * 0.1)
        
        # 3. 模块复杂度因子（0-0.3）
        involved_blocks = set()
        for pin_id in net.pin_ids:
            pin = self.floorplan.get_pin(pin_id)
            if pin:
                involved_blocks.add(pin.block_id)
        
        module_factor = min(0.3, len(involved_blocks) * 0.05)
        
        total_score = pin_factor + reuse_factor + module_factor
        return min(1.0, total_score)
    
    def _find_connected_components(self) -> List[Tuple[int, List[int]]]:
        """找到连通分量（返回(component_id, nets)元组）"""
        logger.info("寻找连通分量...")
        
        # 构建无向图
        undirected_adj = defaultdict(set)
        
        # 获取所有同构组
        for group_id, group in self.floorplan._isomorphic_groups.items():
            group_pins = list(group.pin_ids)
            involved_nets = set()
            
            for pin_id in group_pins:
                pin = self.floorplan.get_pin(pin_id)
                if pin:
                    involved_nets.add(pin.net_id)
            
            if len(involved_nets) > 1:  # 必须涉及多个net
                involved_nets_list = sorted(list(involved_nets))
                for i in range(len(involved_nets_list)):
                    for j in range(i + 1, len(involved_nets_list)):
                        net1, net2 = involved_nets_list[i], involved_nets_list[j]
                        undirected_adj[net1].add(net2)
                        undirected_adj[net2].add(net1)
        
        logger.info(f"无向图构建完成：{len(undirected_adj)}个节点有连接")
        
        # BFS找到连通分量
        visited = set()
        components = []
        
        for net_id in range(self.floorplan.num_nets):
            if net_id in visited:
                continue
                
            component_nets = []
            queue = deque([net_id])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                    
                visited.add(current)
                component_nets.append(current)
                
                for neighbor in undirected_adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if component_nets:
                components.append((len(components), sorted(component_nets)))  # (id, nets)
        
        logger.info(f"连通分量分解完成：{len(components)}个分量")
        return components
    
    def _build_isomorphic_groups(self):
        """构建同构组详细信息，包括为未被覆盖的nets创建虚拟同构组"""
        logger.info("构建同构组信息...")
        
        # 首先收集所有被现有同构组覆盖的nets
        covered_nets = set()
        
        for group_id, group in self.floorplan._isomorphic_groups.items():
            group_pins = list(group.pin_ids)
            involved_nets = set()
            
            # 收集涉及的nets
            for pin_id in group_pins:
                pin = self.floorplan.get_pin(pin_id)
                if pin:
                    involved_nets.add(pin.net_id)
            
            if len(involved_nets) <= 1:  # 只涉及单个net，跳过
                continue
            
            # 计算优先级分数（基于net分数的平均值）
            avg_score = sum(self.net_scores.get(net_id, 0) for net_id in involved_nets) / len(involved_nets)
            
            # 确定核心Pin（优先级最高的Pin）
            core_pin_id = None
            best_score = -1
            for pin_id in group_pins:
                pin = self.floorplan.get_pin(pin_id)
                if pin:
                    net_id = pin.net_id
                    score = self.net_scores.get(net_id, 0)
                    if score > best_score:
                        best_score = score
                        core_pin_id = pin_id
            
            self.isomorphic_groups[group_id] = IsomorphicGroupInfo(
                group_id=group_id,
                pin_ids=group_pins,
                involved_nets=involved_nets,
                priority_score=avg_score,
                core_pin_id=core_pin_id
            )
            
            # 标记这些nets为已覆盖
            covered_nets.update(involved_nets)
        
        logger.info(f"同构组信息构建完成：{len(self.isomorphic_groups)}个有效组，覆盖{len(covered_nets)}个nets")
    
    def _determine_isomorphic_group_order(self):
        """确定同构组处理顺序"""
        logger.info("确定同构组处理顺序...")
        
        # 按优先级分数排序（分数高的先处理）
        sorted_groups = sorted(self.isomorphic_groups.values(), 
                             key=lambda x: x.priority_score, reverse=True)
        
        for i, group_info in enumerate(sorted_groups):
            group_info.processing_order = i
            logger.debug(f"同构组{group_info.group_id}: 优先级={group_info.priority_score:.2f}, 顺序={i}, 核心Pin={group_info.core_pin_id}")
        
        logger.info(f"同构组处理顺序确定完成：{len(sorted_groups)}个组")
    
    def _generate_processing_order(self, components: List[Tuple[int, List[int]]]) -> List[ProcessingUnit]:
        """生成精细处理顺序（按连通分量net数排序）"""
        logger.info("生成精细处理顺序（按连通分量net数排序）...")
        
        processing_units = []
        
        # 按连通分量net数排序（从多到少）
        sorted_components = sorted(components,
                                 key=lambda x: len(x[1]), reverse=True)
        
        logger.info(f"连通分量排序完成：按net数从多到少排序")
        
        for sort_order, (original_component_id, component_nets) in enumerate(sorted_components):
            logger.info(f"处理连通分量 {original_component_id} (排序{sort_order}): {len(component_nets)}个nets")
            
            component_net_set = set(component_nets)
            
            # 追踪该连通分量中已经被处理的nets
            processed_nets = set()
            
            # 获取该连通分量内的所有同构组
            component_groups = []
            for group_info in self.isomorphic_groups.values():
                # 检查该同构组是否主要涉及这个连通分量
                group_nets_in_component = group_info.involved_nets.intersection(component_net_set)
                
                if len(group_nets_in_component) > 1:  # 在该分量内有多个net
                    # 计算该组在该分量内的优先级
                    avg_score = sum(self.net_scores.get(net_id, 0) for net_id in group_nets_in_component) / len(group_nets_in_component)
                    
                    component_groups.append({
                        'group_info': group_info,
                        'local_nets': sorted(list(group_nets_in_component)),
                        'local_score': avg_score
                    })
                    # 标记这些nets为已处理
                    processed_nets.update(group_nets_in_component)
            
            # 按优先级排序同构组
            component_groups.sort(key=lambda x: x['local_score'], reverse=True)
            
            # 为每个同构组生成处理单元
            for stage, group_data in enumerate(component_groups):
                group_info = group_data['group_info']
                local_nets = group_data['local_nets']
                
                # 按net优先级排序
                scored_nets = [(net_id, self.net_scores.get(net_id, 0)) for net_id in local_nets]
                scored_nets.sort(key=lambda x: x[1], reverse=True)
                
                sorted_net_ids = [net_id for net_id, _ in scored_nets]
                
                processing_unit = ProcessingUnit(
                    component_id=original_component_id,
                    component_size=len(component_nets),
                    component_sort_order=sort_order,
                    isomorphic_group_id=group_info.group_id,
                    net_ids=sorted_net_ids,
                    is_reuse_critical=True,  # 跨net的同构组都是复用关键的
                    core_pin_id=group_info.core_pin_id,
                    processing_stage=stage,
                    priority_score=group_data['local_score']
                )
                
                processing_units.append(processing_unit)
                
                logger.debug(f"  处理单元(同构组): 分量{original_component_id}(排序{sort_order}), 同构组{group_info.group_id}, nets={sorted_net_ids}, 核心Pin={group_info.core_pin_id}")
            
            # 处理该连通分量中未被任何同构组覆盖的nets（包括仅有1个net的情况）
            unprocessed_nets = component_net_set - processed_nets
            if unprocessed_nets:
                logger.info(f"  连通分量{original_component_id}中有{len(unprocessed_nets)}个nets未被同构组覆盖，使用最后一个pin的同构组")
                
                # 为每个未处理的net单独创建处理单元
                for net_id in sorted(unprocessed_nets):
                    # 获取该net的最后一个pin
                    net = self.floorplan.get_net(net_id)
                    if not net or not net.pin_ids:
                        logger.warning(f"  net {net_id}没有pins，跳过")
                        continue
                    
                    last_pin_id = net.pin_ids[-1]
                    last_pin = self.floorplan.get_pin(last_pin_id)
                    
                    if not last_pin:
                        logger.warning(f"  无法获取net {net_id}的最后一个pin {last_pin_id}，跳过")
                        continue
                    
                    # 使用最后一个pin的同构组ID（如果有的话）
                    pin_isomorphic_group_id = getattr(last_pin, 'isomorphic_group_id', None)
                    
                    # 如果该pin有同构组，使用同构组的core_pin_id，否则使用该pin本身
                    core_pin_id = last_pin_id
                    if pin_isomorphic_group_id is not None and pin_isomorphic_group_id in self.isomorphic_groups:
                        group_info = self.isomorphic_groups[pin_isomorphic_group_id]
                        core_pin_id = group_info.core_pin_id
                    
                    processing_unit = ProcessingUnit(
                        component_id=original_component_id,
                        component_size=len(component_nets),
                        component_sort_order=sort_order,
                        isomorphic_group_id=pin_isomorphic_group_id if pin_isomorphic_group_id is not None else -1,
                        net_ids=[net_id],
                        is_reuse_critical=False,  # 单net处理单元不复用关键
                        core_pin_id=core_pin_id,
                        processing_stage=len(component_groups),  # 排在同构组之后
                        priority_score=self.net_scores.get(net_id, 0)
                    )
                    
                    processing_units.append(processing_unit)
                    logger.debug(f"  处理单元(单net): 分量{original_component_id}(排序{sort_order}), net={net_id}, 同构组={pin_isomorphic_group_id}, 核心Pin={core_pin_id}")
        
        logger.info(f"精细处理顺序生成完成：{len(processing_units)}个处理单元")
        return processing_units
    
    def _print_refined_stats(self, processing_units: List[ProcessingUnit]):
        """打印精细统计信息"""
        if not processing_units:
            logger.info("没有生成处理单元")
            return
        
        total_units = len(processing_units)
        reuse_critical_units = [unit for unit in processing_units if unit.is_reuse_critical]
        
        logger.info("=== 精细拓扑排序统计 ===")
        logger.info(f"总处理单元: {total_units}")
        logger.info(f"复用关键单元: {len(reuse_critical_units)}")
        logger.info(f"普通单元: {total_units - len(reuse_critical_units)}")
        
        # 按连通分量分组统计
        component_stats = defaultdict(list)
        for unit in processing_units:
            component_stats[unit.component_id].append(unit)
        
        logger.info("连通分量分布:")
        for component_id, units in sorted(component_stats.items()):
            reuse_critical_count = sum(1 for unit in units if unit.is_reuse_critical)
            avg_size = sum(len(unit.net_ids) for unit in units) / len(units) if units else 0
            # logger.info(f"  连通分量{component_id}: {len(units)}个处理单元, {reuse_critical_count}个复用关键, 平均{avg_size:.1f}个nets/单元")
        
        # 同构组大小分布
        size_dist = defaultdict(int)
        for unit in processing_units:
            size = len(unit.net_ids)
            size_dist[size] += 1
        
        logger.info("同构组大小分布:")
        for size in sorted(size_dist.keys()):
            count = size_dist[size]
            logger.info(f"  大小{size}: {count}个")
        
        # 连通分量大小分布
        component_size_dist = defaultdict(int)
        for unit in processing_units:
            component_size_dist[unit.component_size] += 1
        
        logger.info("连通分量大小分布:")
        for size in sorted(component_size_dist.keys()):
            count = component_size_dist[size]
            logger.info(f"  大小{size}: {count}个分量")
        
        logger.info("===================")

def test_refined_topology():
    """测试精细拓扑排序"""
    logger.info("=== 测试精细拓扑排序 ===")
    
    # 加载测试数据
    base_dir = "/root/autodl-tmp/pin_assignment"
    block_json_path = os.path.join(base_dir, "test_case_from_huawei/block_case2.json")
    pingroup_json_path = os.path.join(base_dir, "test_case_from_huawei/pingroup_case2.json")
    
    logger.info("加载测试数据...")
    from PlaceDB import PlaceDB
    
    placedb = PlaceDB(block_json_path, pingroup_json_path)
    floorplan = placedb.convert_to_floorplan_ro()
    
    logger.info(f"数据加载完成: {floorplan.num_nets}个nets, {placedb.total_pin_count}个pins")
    
    # 构建精细拓扑
    logger.info("构建精细拓扑...")
    builder = FinalTopologyBuilder(floorplan)
    processing_units = builder.build_final_topology()
    
    logger.info("=== 精细拓扑测试完成 ===")
    
    return processing_units

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== 精细拓扑排序测试 ===")
    processing_units = test_refined_topology()
    print("=== 测试完成 ===")
    
    # 打印关键结果
    print(f"\n关键结果:")
    print(f"  处理单元数量: {len(processing_units)}")
    if processing_units:
        print(f"  第一个处理单元: {processing_units[0]}")
        reuse_critical_count = sum(1 for unit in processing_units if unit.is_reuse_critical)
        print(f"  复用关键单元: {reuse_critical_count}")
        print(f"  普通单元: {len(processing_units) - reuse_critical_count}")