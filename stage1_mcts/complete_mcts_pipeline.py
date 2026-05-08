#!/usr/bin/env python3
"""
完整的多网同构引脚分配MCTS搜索流程
基于现有的FinalTopologyBuilder和PlaceDB转换函数
从华为真实数据到最终分配结果的完整可执行脚本
"""

import json
import logging
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

# 导入现有的算法模块
from PlaceDB import PlaceDB
from data_model import FloorPlanRO, IsomorphicPinGroup, Segment, MasterBlock, Block, Pin, Net, PinAssignment, SegmentUsage
from final_topology import FinalTopologyBuilder, ProcessingUnit
from simple_reward_function import SimpleRewardFunction
from simple_isomorphic_mcts import SimplifiedIsomorphicMCTS
from multi_net_isomorphic_mcts import MultiNetIsomorphicMCTS, MCTSConfig, AssignmentResult
from segment_usages_direct_interface import SegmentUsagesDirectInterface, create_segment_usages_direct_interface

# 导入 feedthrough 相关模块
from feedthrough import ftpred_loader

# 导入线长计算模块
import importlib.util
spec = importlib.util.spec_from_file_location("calculate_wirelength",
    os.path.join(os.path.dirname(__file__), "calculate_wirelength.py"))
calculate_wirelength_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculate_wirelength_module)

# 配置日志
_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'complete_mcts_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class CompleteMCTSPipeline:
    """完整MCTS搜索流程执行器"""
    
    def __init__(self, ftpred_path: str = None):
        self.placedb = None
        self.floorplan = None
        self.processing_units = []
        self.isomorphic_groups = []
        self.assignment_results = {}
        self.search_statistics = {}
        
        # 全局状态管理 - 跟踪分配过程中的状态变化
        self.global_segment_usages = {}  # segment_id -> SegmentUsage
        self.global_net_assignments = {}  # net_id -> List[PinAssignment]
        self.assigned_isomorphic_groups = set()  # 已分配的同构组ID
        
        # Feedthrough 计算器（常驻会话）
        self.ftpred_session = None
        self.ftpred_modules = None
        self.ftpred_path = ftpred_path or self._default_ftpred_path()
    
    def _default_ftpred_path(self) -> str:
        """获取默认的 ftpred 可执行文件路径"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "feedthrough", "build", "ftpred.exe" if os.name == "nt" else "ftpred")
    
    def _init_ftpred_session(self):
        """初始化 feedthrough 计算器（在 placedb 加载后调用）"""
        if self.placedb is None:
            logger.error("无法初始化 feedthrough：placedb 未加载")
            return False
        
        try:
            # 构建 modules（一次）
            self.ftpred_modules = ftpred_loader.build_modules_text(self.placedb)
            # 创建常驻会话
            self.ftpred_session = ftpred_loader.FtpredBinSession(self.ftpred_path, self.ftpred_modules)
            logger.info(f"Feedthrough 计算器初始化成功，路径: {self.ftpred_path}")
            return True
        except Exception as e:
            logger.error(f"Feedthrough 计算器初始化失败: {e}")
            self.ftpred_session = None
            return False
    
    def close(self):
        """关闭资源"""
        if self.ftpred_session is not None:
            try:
                self.ftpred_session.close()
                logger.info("Feedthrough 会话已关闭")
            except Exception as e:
                logger.error(f"关闭 Feedthrough 会话失败: {e}")
    
    def run_complete_mcts_pipeline(self, block_file: str, pingroup_file: str, 
                                 num_simulations: int = 100, time_limit: float = 1000) -> bool:
        """运行完整的MCTS搜索分配流程"""
        
        logger.info("="*80)
        logger.info("开始运行完整的多网同构引脚分配MCTS搜索流程")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # 1. 加载华为数据并转换为我们的格式
            logger.info("步骤1: 加载华为数据并转换为FloorPlanRO格式")
            if not self._load_and_convert_data(block_file, pingroup_file):
                logger.error("数据加载和转换失败")
                return False
            
            # 2. 构建精细拓扑排序
            logger.info("步骤2: 构建精细拓扑排序")
            if not self._build_refined_topology():
                logger.error("拓扑构建失败")
                return False
            
            # 3. 为每个处理单元运行MCTS搜索
            logger.info(f"步骤3: 为{len(self.processing_units)}个处理单元运行MCTS搜索")
            if not self._run_mcts_search(num_simulations, time_limit):
                logger.error("MCTS搜索失败")
                return False
            
            # 4. 汇总和分析结果
            logger.info("步骤4: 汇总和分析分配结果")
            self._summarize_results()
            
            total_time = time.time() - start_time
            logger.info(f"完整MCTS流程完成！总用时: {total_time:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"完整MCTS流程执行失败: {e}", exc_info=True)
            return False
    
    def _load_and_convert_data(self, block_file: str, pingroup_file: str) -> bool:
        """加载和转换数据"""
        
        try:
            logger.info(f"开始加载华为数据: {block_file}, {pingroup_file}")
            
            # 创建PlaceDB并加载数据
            self.placedb = PlaceDB(block_file, pingroup_file)
            logger.info(f"PlaceDB创建完成: {len(self.placedb.all_modules_list)} 模块, {len(self.placedb.nets_list)} 网表, {self.placedb.total_pin_count} 引脚")
            
            # 转换为FloorPlanRO格式
            self.floorplan = self.placedb.convert_to_floorplan_ro()
            logger.info(f"FloorPlanRO转换完成: {self.floorplan.num_segments} segments, {self.floorplan.num_nets} nets")
            
            # 提取同构组信息
            self.isomorphic_groups = list(self.floorplan._isomorphic_groups.values())
            logger.info(f"同构组提取完成: {len(self.isomorphic_groups)} 组")
            
            # 初始化 feedthrough 计算器（在 placedb 加载后）
            self._init_ftpred_session()
            
            return True
            
        except Exception as e:
            logger.error(f"数据加载和转换失败: {e}", exc_info=True)
            return False
    
    def _build_refined_topology(self) -> bool:
        """构建精细拓扑排序"""
        
        try:
            logger.info("构建精细拓扑排序...")
            
            # 使用现有的FinalTopologyBuilder
            topology_builder = FinalTopologyBuilder(self.floorplan)
            self.processing_units = topology_builder.build_final_topology()
            
            logger.info(f"精细拓扑构建完成: {len(self.processing_units)} 个处理单元")
            
            # 打印统计信息
            # self._print_topology_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"拓扑构建失败: {e}", exc_info=True)
            return False
    
    def _run_mcts_search(self, num_simulations: int, time_limit: float) -> bool:
        """运行MCTS搜索"""
        
        try:
            logger.info(f"开始MCTS搜索: 模拟次数={num_simulations}, 时间限制={time_limit}秒")
            
            total_results = []
            successful_assignments = 0
            failed_assignments = 0
            
            # 按处理单元顺序进行分配
            for i, processing_unit in enumerate(self.processing_units):
                logger.info(f"处理单元 {i+1}/{len(self.processing_units)}: "
                          f"同构组={processing_unit.isomorphic_group_id}, "
                          f"nets={len(processing_unit.net_ids)}, "
                          f"分量={processing_unit.component_id}")
                
                if processing_unit.component_id == 69:
                    logger.debug(f"调试信息: 处理单元{i+1}属于连通分量{processing_unit.component_id}")
                    logger.debug(f"同构组ID: {processing_unit.isomorphic_group_id}, 包含nets: {processing_unit.net_ids}")
                
                # 获取该处理单元对应的同构组
                group_id = processing_unit.isomorphic_group_id
                isomorphic_group = self.floorplan._isomorphic_groups.get(group_id)
                
                if not isomorphic_group:
                    logger.warning(f"找不到同构组 {group_id}")
                    failed_assignments += 1
                    continue
                
                # 配置MCTS参数
                config = MCTSConfig(
                    num_simulations=num_simulations,
                    time_limit=time_limit / len(self.processing_units),  # 平均分配时间
                    allocation_strategy="adaptive",
                    enable_backtrack=True
                )
                
                # # 创建MCTS分配器 - 传入当前全局状态
                # constraint_manager = SmartIsomorphicConstraintManager(
                #     floorplan=self.floorplan,
                #     allocation_strategy=AllocationStrategy.ADAPTIVE
                # )
                
                reward_function = SimpleRewardFunction(
                    floorplan=self.floorplan,
                    ftpred_session=self.ftpred_session,
                    placedb=self.placedb
                )
                
                mcts_allocator = MultiNetIsomorphicMCTS(
                    floorplan=self.floorplan,
                    config=config,
                    initial_segment_usages=self.global_segment_usages,
                    initial_net_assignments=self.global_net_assignments
                )
                
                # 运行分配
                start_unit_time = time.time()
                result = mcts_allocator.assign_isomorphic_group(isomorphic_group)
                unit_time = time.time() - start_unit_time
                
                if result.success:
                    logger.info(f"✓ 处理单元 {i+1} 分配成功:")
                    logger.info(f"  约束满足率: {result.constraint_satisfaction_rate:.2%}")
                    logger.info(f"  总线长: {result.total_wirelength:.2f}")
                    logger.info(f"  用时: {unit_time:.2f}秒")
                    
                    # 更新全局状态 - 应用分配结果
                    self._update_global_state_from_result(result, isomorphic_group)
                    
                    successful_assignments += 1
                    total_results.append({
                        'unit_id': i,
                        'group_id': group_id,
                        'success': True,
                        'constraint_satisfaction': result.constraint_satisfaction_rate,
                        'wirelength': result.total_wirelength,
                        'assignments': result.assignments,
                        'execution_time': unit_time
                    })
                else:
                    logger.error(f"✗ 处理单元 {i+1} 分配失败: {result.error_message}")
                    failed_assignments += 1
                    total_results.append({
                        'unit_id': i,
                        'group_id': group_id,
                        'success': False,
                        'error': result.error_message,
                        'execution_time': unit_time
                    })
            
            # 保存结果
            self.assignment_results = {
                'total_units': len(self.processing_units),
                'successful_assignments': successful_assignments,
                'failed_assignments': failed_assignments,
                'success_rate': successful_assignments / len(self.processing_units) if self.processing_units else 0.0,
                'detailed_results': total_results
            }
            
            logger.info(f"MCTS搜索完成:")
            logger.info(f"  总处理单元: {len(self.processing_units)}")
            logger.info(f"  成功分配: {successful_assignments}")
            logger.info(f"  失败分配: {failed_assignments}")
            logger.info(f"  成功率: {self.assignment_results['success_rate']:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"MCTS搜索失败: {e}", exc_info=True)
            return False
    
    def _summarize_results(self):
        """汇总和分析结果"""
        
        if not self.assignment_results:
            logger.warning("没有分配结果可汇总")
            return
        
        results = self.assignment_results
        successful_results = [r for r in results['detailed_results'] if r['success']]
        
        logger.info("="*60)
        logger.info("分配结果汇总:")
        logger.info(f"总处理单元: {results['total_units']}")
        logger.info(f"成功分配: {results['successful_assignments']}")
        logger.info(f"失败分配: {results['failed_assignments']}")
        logger.info(f"整体成功率: {results['success_rate']:.2%}")
        
        if successful_results:
            avg_constraint_satisfaction = sum(r['constraint_satisfaction'] for r in successful_results) / len(successful_results)
            total_wirelength = sum(r['wirelength'] for r in successful_results)
            avg_execution_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
            
            logger.info(f"平均约束满足率: {avg_constraint_satisfaction:.2%}")
            logger.info(f"总线长: {total_wirelength:.2f}")
            logger.info(f"平均执行时间: {avg_execution_time:.2f}秒/单元")
        
        # 按同构组大小分析
        if successful_results:
            size_analysis = defaultdict(list)
            for result in successful_results:
                group_id = result['group_id']
                group = self.floorplan._isomorphic_groups.get(group_id)
                if group:
                    size = len(group.pin_ids)
                    size_analysis[size].append(result['constraint_satisfaction'])
            
            if size_analysis:
                logger.info("同构组大小分析:")
                for size in sorted(size_analysis.keys()):
                    count = len(size_analysis[size])
                    avg_satisfaction = sum(size_analysis[size]) / count
                    logger.info(f"  大小{size}: {count}组, 平均约束满足率{avg_satisfaction:.2%}")
        
        logger.info("="*60)
    
    def _print_topology_stats(self):
        """打印拓扑统计信息"""
        
        if not self.processing_units:
            logger.info("没有处理单元可统计")
            return
        
        total_units = len(self.processing_units)
        reuse_critical_units = [unit for unit in self.processing_units if unit.is_reuse_critical]
        
        logger.info("=== 拓扑统计信息 ===")
        logger.info(f"总处理单元: {total_units}")
        logger.info(f"复用关键单元: {len(reuse_critical_units)}")
        logger.info(f"普通单元: {total_units - len(reuse_critical_units)}")
        
        # 按连通分量分组统计
        component_stats = defaultdict(list)
        for unit in self.processing_units:
            component_stats[unit.component_id].append(unit)
        
        logger.info("连通分量分布:")
        for component_id, units in sorted(component_stats.items()):
            reuse_critical_count = sum(1 for unit in units if unit.is_reuse_critical)
            avg_size = sum(len(unit.net_ids) for unit in units) / len(units) if units else 0
            logger.info(f"  连通分量{component_id}: {len(units)}个处理单元, {reuse_critical_count}个复用关键, 平均{avg_size:.1f}个nets/单元")
        
        # 同构组大小分布
        size_dist = defaultdict(int)
        for unit in self.processing_units:
            size = len(unit.net_ids)
            size_dist[size] += 1
        
        logger.info("同构组大小分布:")
        for size in sorted(size_dist.keys()):
            count = size_dist[size]
            logger.info(f"  大小{size}: {count}个")
        
        logger.info("===================")
    
    def _update_global_state_from_result(self, result: AssignmentResult, isomorphic_group: IsomorphicPinGroup):
        """从分配结果更新全局状态"""
        
        if not result.success:
            return
        
        logger.info(f"更新全局状态: 同构组{isomorphic_group.group_id}的分配结果")
        
        # 1. 更新segment使用情况 - 使用更新方式而非直接添加
        for net_id, assignments in result.assignments.items():
            for pin_id, seg_id in assignments:
                # 获取或创建segment使用记录
                if seg_id not in self.global_segment_usages:
                    self.global_segment_usages[seg_id] = SegmentUsage(seg_id=seg_id)
                
                segment_usage = self.global_segment_usages[seg_id]
                
                # 检查是否已存在该pin的分配
                existing_assignment = None
                for existing_net_id, existing_assignments in self.global_net_assignments.items():
                    for existing_assignment_obj in existing_assignments:
                        if (hasattr(existing_assignment_obj, 'pin_id') and
                            existing_assignment_obj.pin_id == pin_id):
                            existing_assignment = existing_assignment_obj
                            break
                    if existing_assignment:
                        break
                
                if existing_assignment and existing_assignment.seg_id != -1 and seg_id != existing_assignment.seg_id:
                    # 如果已存在实际分配，先减少旧segment的使用
                    if existing_assignment.seg_id in self.global_segment_usages:
                        old_usage = self.global_segment_usages[existing_assignment.seg_id]
                        if pin_id in old_usage.assigned_pins:
                            old_usage.assigned_pins.remove(pin_id)
                            # 减少容量使用
                            pin_obj = self.floorplan.get_pin(pin_id)
                            if pin_obj and hasattr(pin_obj, 'width'):
                                old_usage.used_capacity -= pin_obj.width
                                logger.debug(f"  减少旧segment{existing_assignment.seg_id}容量: {pin_obj.width}")
                
                # TODO:同构Pin遇到第一个pin就完成了分配，后续的同构Pin没有分配到segment上，当前seg分配到上限容量时会出现bug
                # 分配pin到新segment
                segment_usage.assign_pin(pin_id, self.floorplan)
                logger.debug(f"  分配pin {pin_id} 到segment {seg_id}")
        
        # 2. 更新网表分配记录 - 使用更新方式而非直接添加
        for net_id, assignments in result.assignments.items():
            if net_id not in self.global_net_assignments:
                self.global_net_assignments[net_id] = []
            
            for pin_id, seg_id in assignments:
                # 检查是否已存在该pin的分配
                existing_assignment = None
                existing_index = -1
                
                for i, existing in enumerate(self.global_net_assignments[net_id]):
                    if hasattr(existing, 'pin_id') and existing.pin_id == pin_id:
                        existing_assignment = existing
                        existing_index = i
                        break
                
                new_assignment = PinAssignment(pin_id=pin_id, seg_id=seg_id, net_id=net_id)
                
                if existing_assignment:
                    # 更新现有分配
                    logger.debug(f"  更新分配: pin{pin_id} 从seg{existing_assignment.seg_id} 到seg{seg_id}")
                    self.global_net_assignments[net_id][existing_index] = new_assignment
                else:
                    # 添加新分配
                    logger.debug(f"  添加新分配: pin{pin_id} -> seg{seg_id}")
                    self.global_net_assignments[net_id].append(new_assignment)
        
        # 3. 记录已分配的同构组
        self.assigned_isomorphic_groups.add(isomorphic_group.group_id)
        
        logger.info(f"全局状态更新完成: 已分配 {len(result.assignments)} 个网表, "
                   f"更新了 {len(self.global_segment_usages)} 个segment的使用情况")
    
    def calculate_wirelength(self, assignment_file: str, pingroup_file: str, block_file: str) -> dict:
        """
        计算线长和 feedthrough
        
        参数:
            assignment_file: 分配结果文件路径
            pingroup_file: pingroup文件路径
            block_file: block文件路径
        
        返回:
            包含线长统计信息的字典
        """
        logger.info("=" * 60)
        logger.info("开始计算线长和 Feedthrough")
        logger.info("=" * 60)
        
        try:
            # 使用 calculate_wirelength 模块的功能
            # 1. 加载数据
            logger.info(f"加载分配结果: {assignment_file}")
            assignment_data = calculate_wirelength_module.load_json(assignment_file)
            pingroup_data = calculate_wirelength_module.load_json(pingroup_file)
            block_data = calculate_wirelength_module.load_json(block_file)
            
            # 2. 解析数据
            pin_assignments = calculate_wirelength_module.parse_assignments(assignment_data)
            blocks_info = calculate_wirelength_module.parse_block_vertices(block_data)
            pin_name_to_id = calculate_wirelength_module.build_pin_name_to_id_map(pin_assignments)
            
            # 3. 初始化 feedthrough 计算器
            placedb, ftpred_session = None, None
            if self.placedb is not None and self.ftpred_session is not None:
                placedb = self.placedb
                ftpred_session = self.ftpred_session
                logger.info("使用现有的 feedthrough 计算器")
            else:
                logger.info("初始化 feedthrough 计算器...")
                placedb, ftpred_session = calculate_wirelength_module.init_feedthrough_calculator(
                    block_file, pingroup_file
                )
            
            # 4. 计算每个 net 的线长
            nets = calculate_wirelength_module.parse_pingroup_nets(pingroup_data)
            missing_pins = set()
            net_results = []
            total_hpwl = 0.0
            total_feedthrough = 0
            successful_nets = 0
            failed_nets = 0
            skipped_single_pin_nets = 0
            feedthrough_failed_nets = 0
            
            for net_idx, net_pingroup_chain in enumerate(nets):
                hpwl, locations, skipped = calculate_wirelength_module.calculate_net_wirelength(
                    net_pingroup_chain, pin_assignments, pin_name_to_id, blocks_info, missing_pins,
                    skip_single_pin=True
                )
                
                # 获取 net 名称
                net_name = ""
                if net_pingroup_chain:
                    first_pingroup = net_pingroup_chain[0]
                    successors = first_pingroup.get("successors", [])
                    if successors:
                        net_name = successors[0]
                    else:
                        net_name = first_pingroup.get("pingroup_name", f"net_{net_idx}")
                
                if skipped:
                    skipped_single_pin_nets += 1
                    net_results.append({
                        "net_index": net_idx,
                        "net_name": net_name,
                        "pingroup_count": len(net_pingroup_chain),
                        "found_pins": 0,
                        "hpwl": 0.0,
                        "skipped": True,
                        "skip_reason": "single_pin_net",
                        "locations": []
                    })
                elif locations:
                    total_hpwl += hpwl
                    successful_nets += 1
                    
                    # 计算 feedthrough
                    feedthrough = -1
                    if ftpred_session is not None and placedb is not None:
                        feedthrough = calculate_wirelength_module.calculate_net_feedthrough(
                            net_pingroup_chain, locations, placedb, ftpred_session
                        )
                        if feedthrough >= 0:
                            total_feedthrough += feedthrough
                        else:
                            feedthrough_failed_nets += 1
                    
                    net_results.append({
                        "net_index": net_idx,
                        "net_name": net_name,
                        "pingroup_count": len(net_pingroup_chain),
                        "found_pins": len(locations),
                        "hpwl": hpwl,
                        "feedthrough": feedthrough if feedthrough >= 0 else None,
                        "skipped": False,
                        "locations": [{"pin_name": loc.pin_name, "x": loc.x, "y": loc.y,
                                      "source": loc.source, "block_name": loc.block_name}
                                     for loc in locations]
                    })
                else:
                    failed_nets += 1
                    net_results.append({
                        "net_index": net_idx,
                        "net_name": net_name,
                        "pingroup_count": len(net_pingroup_chain),
                        "found_pins": 0,
                        "hpwl": 0.0,
                        "feedthrough": None,
                        "skipped": False,
                        "locations": []
                    })
            
            # 5. 输出结果统计
            logger.info("\n线长和 Feedthrough 计算结果")
            logger.info("=" * 60)
            
            valid_nets = successful_nets + failed_nets
            logger.info(f"\n统计信息:")
            logger.info(f"   - 总net数: {len(nets)}")
            logger.info(f"   - 单pin net (跳过): {skipped_single_pin_nets}")
            logger.info(f"   - 有效net数 (>=2 pins): {valid_nets}")
            logger.info(f"   - 成功计算线长的net: {successful_nets}")
            logger.info(f"   - 未找到任何pin的net: {failed_nets}")
            logger.info(f"   - 找不到的pin数: {len(missing_pins)}")
            logger.info(f"   - 总HPWL线长: {total_hpwl:.2f}")
            logger.info(f"   - 平均HPWL线长: {total_hpwl / successful_nets if successful_nets > 0 else 0:.2f}")
            
            if ftpred_session is not None:
                successful_feedthrough_nets = successful_nets - feedthrough_failed_nets
                logger.info(f"   - Feedthrough 计算成功net数: {successful_feedthrough_nets}")
                logger.info(f"   - Feedthrough 计算失败net数: {feedthrough_failed_nets}")
                logger.info(f"   - 总Feedthrough数: {total_feedthrough}")
                logger.info(f"   - 平均Feedthrough数: {total_feedthrough / successful_feedthrough_nets if successful_feedthrough_nets > 0 else 0:.2f}")
            
            # 6. 保存详细结果
            result_dir = Path("mcts_result")
            wirelength_output_file = result_dir / "wirelength_results.json"
            with open(wirelength_output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "total_nets": len(nets),
                        "successful_nets": successful_nets,
                        "failed_nets": failed_nets,
                        "missing_pins_count": len(missing_pins),
                        "total_hpwl": total_hpwl,
                        "average_hpwl": total_hpwl / successful_nets if successful_nets > 0 else 0,
                        "total_feedthrough": total_feedthrough if ftpred_session else None,
                        "average_feedthrough": (total_feedthrough / (successful_nets - feedthrough_failed_nets)
                                               if ftpred_session and (successful_nets - feedthrough_failed_nets) > 0 else None)
                    },
                    "net_results": net_results,
                    "missing_pins": sorted(list(missing_pins))
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"\n详细结果已保存到: {wirelength_output_file}")
            
            # 输出找不到的 pin 列表
            if missing_pins:
                missing_pins_file = result_dir / "missing_pins.txt"
                with open(missing_pins_file, 'w', encoding='utf-8') as f:
                    f.write(f"找不到的Pin列表 (共{len(missing_pins)}个):\n")
                    f.write("=" * 60 + "\n")
                    for pin_name in sorted(missing_pins):
                        f.write(f"{pin_name}\n")
                logger.info(f"找不到的pin列表已保存到: {missing_pins_file}")
            
            # 输出前20个net的详情
            logger.info("\n前20个net的线长详情:")
            logger.info("-" * 100)
            if ftpred_session:
                logger.info(f"{'Net名称':<50} {'Pins':>6} {'HPWL':>12} {'Feedthrough':>12}")
            else:
                logger.info(f"{'Net名称':<50} {'Pins':>6} {'HPWL':>12}")
            logger.info("-" * 100)
            
            count = 0
            for result in net_results:
                if result["hpwl"] > 0 and count < 20:
                    feedthrough_str = ""
                    if ftpred_session and result.get("feedthrough") is not None:
                        feedthrough_str = f"{result['feedthrough']:>12}"
                    
                    if ftpred_session:
                        logger.info(f"{result['net_name'][:49]:<50} {result['found_pins']:>6} {result['hpwl']:>12.2f} {feedthrough_str}")
                    else:
                        logger.info(f"{result['net_name'][:49]:<50} {result['found_pins']:>6} {result['hpwl']:>12.2f}")
                    count += 1
            
            logger.info("=" * 60)
            logger.info("线长计算完成!")
            
            # 7. 修复缺失的 pin 分配
            if missing_pins and placedb is not None:
                logger.info("\n开始修复缺失的pin分配...")
                result_dir = Path("mcts_result")
                final_dir = result_dir / "final_result"
                fixed_output_file = final_dir / f"segment_assignments_{int(time.time())}_fixed.json"
                fixed_assignment_data = calculate_wirelength_module.fix_missing_pins(
                    placedb=placedb,
                    assignment_data=assignment_data,
                    missing_pins=missing_pins,
                    pin_name_to_id=pin_name_to_id,
                    output_file=str(fixed_output_file)
                )
                logger.info(f"修复完成，结果已保存到: {fixed_output_file}")
            
            return {
                "total_nets": len(nets),
                "successful_nets": successful_nets,
                "failed_nets": failed_nets,
                "total_hpwl": total_hpwl,
                "average_hpwl": total_hpwl / successful_nets if successful_nets > 0 else 0,
                "total_feedthrough": total_feedthrough if ftpred_session else None,
                "missing_pins_count": len(missing_pins)
            }
            
        except Exception as e:
            logger.error(f"计算线长时出错: {e}", exc_info=True)
            return {}
    
    def save_results(self, output_file: str):
        """保存结果到文件"""
        
        if not self.assignment_results:
            logger.warning("没有结果可保存")
            return
        
        try:
            # 构建完整的保存数据
            save_data = {
                'metadata': {
                    'timestamp': time.time(),
                    'input_files': {
                        'block_file': getattr(self, 'block_file', 'unknown'),
                        'pingroup_file': getattr(self, 'pingroup_file', 'unknown')
                    },
                    'mcts_config': {
                        'num_simulations': len(self.assignment_results.get('detailed_results', [])),
                        'time_limit': 'distributed'
                    }
                },
                'summary': {
                    'total_processing_units': self.assignment_results['total_units'],
                    'successful_assignments': self.assignment_results['successful_assignments'],
                    'failed_assignments': self.assignment_results['failed_assignments'],
                    'success_rate': self.assignment_results['success_rate']
                },
                'detailed_results': self.assignment_results['detailed_results'],
                'topology_info': {
                    'total_units': len(self.processing_units),
                    'isomorphic_groups': len(self.isomorphic_groups)
                },
                'global_state': {
                    'segment_usages': {
                        seg_id: {
                            'used_capacity': usage.used_capacity,
                            'assigned_pins': usage.assigned_pins,
                            'remaining_capacity': usage.get_remaining_capacity(self.floorplan)
                        } for seg_id, usage in self.global_segment_usages.items()
                    },
                    'net_assignments': {
                        net_id: len(assignments) for net_id, assignments in self.global_net_assignments.items()
                    },
                    'assigned_groups': len(self.assigned_isomorphic_groups)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

def main(block_file=None, pingroup_file=None, num_simulations=1000, time_limit=30.0):
    """主函数 - 支持直接参数输入和命令行参数
    
    Args:
        block_file: Block JSON文件路径
        pingroup_file: PinGroup JSON文件路径
        num_simulations: MCTS模拟次数，默认1000
        time_limit: 时间限制（秒），默认30.0
    """
    
    # 如果没有提供参数，则使用命令行参数
    if block_file is None or pingroup_file is None:
        # 检查命令行参数
        if len(sys.argv) < 3:
            print("使用方法: python complete_mcts_pipeline.py <block.json> <pingroup.json> [num_simulations] [time_limit]")
            print("示例: python complete_mcts_pipeline.py test_case_from_huawei/block.json test_case_from_huawei/pingroup.json 1000 30.0")
            print("或直接调用: main('block.json', 'pingroup.json', 1000, 30.0)")
            sys.exit(1)
        
        block_file = sys.argv[1]
        pingroup_file = sys.argv[2]
        num_simulations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        time_limit = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    
    # 检查文件是否存在
    if not Path(block_file).exists():
        logger.error(f"Block文件不存在: {block_file}")
        sys.exit(1)
    
    if not Path(pingroup_file).exists():
        logger.error(f"PinGroup文件不存在: {pingroup_file}")
        sys.exit(1)
    
    logger.info(f"开始处理: {block_file}, {pingroup_file}")
    logger.info(f"MCTS参数: 模拟次数={num_simulations}, 时间限制={time_limit}秒")
    
    # 创建并运行完整流程
    pipeline = CompleteMCTSPipeline()
    pipeline.block_file = block_file
    pipeline.pingroup_file = pingroup_file
    
    success = pipeline.run_complete_mcts_pipeline(block_file, pingroup_file, num_simulations, time_limit)
    
    if success:
        # 创建结果目录
        result_dir = Path("mcts_result")
        result_dir.mkdir(exist_ok=True)
        
        # 保存详细结果到detailed_info目录
        detailed_dir = result_dir / "detailed_info"
        detailed_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        detailed_output_file = detailed_dir / f"mcts_assignment_results_{timestamp}.json"
        pipeline.save_results(str(detailed_output_file))
        
        logger.info(f"📊 详细结果已保存到: {detailed_output_file}")
        
        # 使用segment_usages_direct_interface处理结果，生成最终格式
        try:
            logger.info("正在使用segment_usages_direct_interface处理结果...")
            
            # 创建接口实例
            interface = create_segment_usages_direct_interface(pipeline.floorplan)
            
            # 从刚保存的详细结果中加载segment_usages
            segment_usages = interface.load_mcts_segment_usages(str(detailed_output_file))
            
            if segment_usages:
                # 转换数据
                final_result = interface.convert_from_mcts_segment_usages(segment_usages)
                
                # 保存最终结果到final_result目录
                final_dir = result_dir / "final_result"
                final_dir.mkdir(exist_ok=True)
                
                final_output_file = final_dir / f"segment_assignments_{timestamp}.json"
                interface.export_segment_assignments(final_result, str(final_output_file))
                
                logger.info(f"🎯 最终segment分配结果已保存到: {final_output_file}")
                
                # 打印一些统计信息
                total_segments = len(final_result)
                
                logger.info(f"📈 统计信息:")
                logger.info(f"  总segments数: {total_segments}")
                
                # 步骤5: 计算线长和 feedthrough
                try:
                    logger.info("\n步骤5: 计算线长和 Feedthrough...")
                    wirelength_stats = pipeline.calculate_wirelength(
                        assignment_file=str(final_output_file),
                        pingroup_file=pingroup_file,
                        block_file=block_file
                    )
                    
                    if wirelength_stats:
                        logger.info(f"✓ 线长计算完成:")
                        logger.info(f"  总HPWL线长: {wirelength_stats.get('total_hpwl', 0):.2f}")
                        logger.info(f"  平均HPWL线长: {wirelength_stats.get('average_hpwl', 0):.2f}")
                        if wirelength_stats.get('total_feedthrough') is not None:
                            logger.info(f"  总Feedthrough: {wirelength_stats.get('total_feedthrough', 0)}")
                    else:
                        logger.warning("线长计算返回空结果")
                        
                except Exception as e:
                    logger.error(f"计算线长时出错: {e}", exc_info=True)
                
            else:
                logger.warning("未能从详细结果中加载segment_usages数据")
                
        except Exception as e:
            logger.error(f"使用segment_usages_direct_interface处理结果时出错: {e}", exc_info=True)
        
        logger.info("="*80)
        logger.info("🎉 完整MCTS搜索流程执行成功！")
        logger.info("="*80)
        
        sys.exit(0)
    else:
        logger.error("❌ 完整MCTS搜索流程执行失败")
        sys.exit(1)

if __name__ == '__main__':
    main()