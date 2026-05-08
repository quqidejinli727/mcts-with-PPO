"""
PlaceDB.py - 模块信息解析器
用于解析 block.json 文件中的层级模块结构
"""

import json
from enum import IntEnum
from typing import List, Optional, Dict, Any, Tuple
from shapely import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math

# 导入数据模型类
from data_model import Segment, Block, MasterBlock, SegmentInst, Pin, Net, FloorPlanRO, IsomorphicPinGroup


class Pin:
    """
    Pin 类，表示一个引脚
    
    属性:
        parent_inst (str): 所属模块的实例名，如 "TOP.U_J.U_J1"
        parent_module (str): 所属模块的当前层级名称，如 "J1"
        pingroup_name (str): Pin 的名称，如 "src_j1_j2_1"
        scope (List): 作用域（目前为空）
        successors (List[str]): 下一个连接的 Pin，格式为 "parent_inst.pingroup_name"
        width (float): Pin 的宽度
    """
    
    def __init__(
        self,
        parent_inst: str,
        parent_module: str,
        pingroup_name: str,
        scope: List,
        successors: List[str],
        width: float
    ):
        """
        初始化 Pin 对象
        
        Args:
            parent_inst: 所属模块的实例名
            parent_module: 所属模块的当前层级名称
            pingroup_name: Pin 的名称
            scope: 作用域
            successors: 下一个连接的 Pin 列表
            width: Pin 的宽度
        """
        self.parent_inst = parent_inst
        self.parent_module = parent_module
        self.pingroup_name = pingroup_name
        self.scope = scope
        self.successors = successors
        self.width = width
        self.x = 0
        self.y = 0
    
    def get_full_name(self) -> str:
        """
        获取 Pin 的完整名称，格式为 "parent_inst.pingroup_name"
        
        Returns:
            str: Pin 的完整名称
        """
        return f"{self.parent_inst}.{self.pingroup_name}"
    
    def __repr__(self) -> str:
        """返回 Pin 的字符串表示"""
        return f"Pin(parent_inst='{self.parent_inst}', pingroup_name='{self.pingroup_name}', width={self.width})"
    
    def __str__(self) -> str:
        """返回 Pin 的友好字符串表示"""
        return f"{self.get_full_name()} (width={self.width})"


class Net:
    """
    Net 类，表示一个网表，由多个 Pin 组成
    
    属性:
        pins (List[Pin]): 组成该网表的所有 Pin
    """
    
    def __init__(self, pins: List[Pin]):
        """
        初始化 Net 对象
        
        Args:
            pins: 组成网表的 Pin 列表
        """
        self.pins = pins
        self.net_degree = len(pins)
        self.feedthrough = 0  # 新增属性：feedthrough 计数
    
    def get_pin_count(self) -> int:
        """
        获取网表中 Pin 的数量
        
        Returns:
            int: Pin 的数量
        """
        return len(self.pins)
    
    def __repr__(self) -> str:
        """返回 Net 的字符串表示"""
        return f"Net(pin_count={len(self.pins)})"
    
    def __str__(self) -> str:
        """返回 Net 的友好字符串表示"""
        return f"Net with {len(self.pins)} pins"

class CoordRotation(IntEnum):
    """坐标旋转枚举类，对应 rotate_information.txt"""
    UNDEFINE = -1          # Undefine.
    ROTATION_R0 = 0        # No rotate.
    ROTATION_R180 = 1      # Rotate 180 degree in counterclockwise.
    ROTATION_R90 = 2       # Rotate 90 degree in counterclockwise.
    ROTATION_R270 = 3      # Rotate 270 degree in counterclockwise.
    ROTATION_MY = 4        # Flip symmetrically along Y-axis.
    ROTATION_MX = 5        # Flip symmetrically along X-axis.
    ROTATION_MX90 = 6      # Rotate 90 degree in counterclockwise after flip symmetrically along X-axis.
    ROTATION_MY90 = 7      # Rotate 90 degree in counterclockwise after flip symmetrically along Y-axis.

class Module:
    """
    模块类，表示一个模块及其所有属性信息
    
    属性:
        name (str): 模块的全名，如 "TOP.U_A.U_A1"
        module_name (str): 当前模块的名称，如 "A1"
        vertex (List[List[float]]): 模块的顶点坐标列表，确定模块的位置和形状
        direction (CoordRotation): 旋转方向，表示相较于 direction=0 的对应旋转
        color (str): 模块的颜色，十六进制格式，如 "#ffff00"
        children (List[Module]): 子模块列表
        parent (Optional[Module]): 父模块引用（可选）
        pin_list (List[Pin]): 模块的引脚列表
    """
    
    def __init__(
        self,
        name: str,
        module_name: str,
        vertex: List[List[float]],
        direction: int,
        color: str,
        children: Optional[List['Module']] = None,
        parent: Optional['Module'] = None
    ):
        """
        初始化模块对象
        
        Args:
            name: 模块的全名
            module_name: 当前模块的名称
            vertex: 顶点坐标列表，每个顶点为 [x, y]
            direction: 旋转方向（整数）
            color: 颜色字符串
            children: 子模块列表，默认为空列表
            parent: 父模块引用，默认为 None
        """
        self.name = name
        self.module_name = module_name
        self.vertex = vertex
        self.direction = CoordRotation(direction) if direction >= 0 else CoordRotation.UNDEFINE
        self.color = color
        self.children = children if children is not None else []
        self.parent = parent
        self.polygon = Polygon(vertex)
        self.area = self.polygon.area
        self.pin_list = []
        
        # 设置子模块的父引用
        for child in self.children:
            child.parent = self
    
    def get_hierarchy_level(self) -> int:
        """
        获取模块在层级结构中的层级深度
        
        Returns:
            int: 层级深度，TOP 为 0，其直接子模块为 1，以此类推
        """
        level = 0
        current = self
        while current.parent is not None:
            level += 1
            current = current.parent
        return level
    
    def get_path(self) -> List[str]:
        """
        获取从根模块到当前模块的路径
        
        Returns:
            List[str]: 模块名称路径列表，如 ["TOP", "U_A", "A1"]
        """
        path = []
        current = self
        while current is not None:
            path.insert(0, current.module_name)
            current = current.parent
        return path
    
    def find_module_by_name(self, name: str) -> Optional['Module']:
        """
        在当前模块及其子树中查找指定名称的模块
        
        Args:
            name: 要查找的模块全名
            
        Returns:
            Optional[Module]: 找到的模块对象，如果未找到则返回 None
        """
        if self.name == name:
            return self
        
        for child in self.children:
            result = child.find_module_by_name(name)
            if result is not None:
                return result
        
        return None
    
    def find_modules_by_module_name(self, module_name: str) -> List['Module']:
        """
        在当前模块及其子树中查找所有指定 module_name 的模块
        
        Args:
            module_name: 要查找的模块名称
            
        Returns:
            List[Module]: 找到的所有模块对象列表
        """
        results = []
        if self.module_name == module_name:
            results.append(self)
        
        for child in self.children:
            results.extend(child.find_modules_by_module_name(module_name))
        
        return results
    
    def get_bounding_box(self) -> Dict[str, float]:
        """
        获取模块的边界框（包围盒）
        
        Returns:
            Dict[str, float]: 包含 min_x, max_x, min_y, max_y 的字典
        """
        if not self.vertex:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0}
        
        x_coords = [v[0] for v in self.vertex]
        y_coords = [v[1] for v in self.vertex]
        
        return {
            "min_x": min(x_coords),
            "max_x": max(x_coords),
            "min_y": min(y_coords),
            "max_y": max(y_coords)
        }
    
    def get_area(self) -> float:
        """
        获取模块的面积
        
        Returns:
            float: 模块面积
        """
        return self.area
    
    def is_leaf(self) -> bool:
        """
        判断是否为叶子节点（没有子模块）
        
        Returns:
            bool: 如果是叶子节点返回 True，否则返回 False
        """
        return len(self.children) == 0
    
    def __repr__(self) -> str:
        """返回模块的字符串表示"""
        return f"Module(name='{self.name}', module_name='{self.module_name}', " \
               f"direction={self.direction.name}, children={len(self.children)})"
    
    def __str__(self) -> str:
        """返回模块的友好字符串表示"""
        return f"{self.name} ({self.module_name}) - {self.direction.name}"


class PlaceDB:
    """
    PlaceDB 类，表示一个 PlaceDB 对象
    
    属性:
        root_module: 最顶层的模块，即outline
        all_modules_dict: Dict[str, Module]
            key: module name
            value: module object
        all_modules_list: List[Module]
            list of all modules
        net_list: List[Net]
            list of all nets
        total_pin_count: int
            total number of pins
    """
    def __init__(self, block_json_file_path: str, pingroup_json_file_path: str):
        """
        初始化 PlaceDB 对象
        
        Args:
            block_json_file_path: block.json 文件路径
            pingroup_json_file_path: pingroup.json 文件路径
        """
        self.root_module = self.load_place_db(block_json_file_path)
        self.all_module_dict = {}
        self.all_modules_list = self.collect_all_modules(self.root_module)
        self.total_pin_count = 0
        self.nets_list = self.load_pingroup_json(pingroup_json_file_path)
    
    def __repr__(self) -> str:
        """返回 PlaceDB 的字符串表示"""
        return f"PlaceDB(modules={len(self.all_modules_list)}, nets={len(self.nets_list)}, total_pin_count={self.total_pin_count})"
    
    def __str__(self) -> str:
        """返回 PlaceDB 的友好字符串表示"""
        return f"PlaceDB with {len(self.all_modules_list)} modules , {len(self.nets_list)} nets and {self.total_pin_count} pins"

    def parse_module_from_dict(self, data: Dict[str, Any], parent: Optional[Module] = None) -> Module:
        """
        从字典数据递归解析模块对象
        
        Args:
            data: 包含模块信息的字典
            parent: 父模块对象，默认为 None
            
        Returns:
            Module: 解析后的模块对象
        """
        # 解析子模块
        children = []
        if "children" in data and data["children"]:
            for child_data in data["children"]:
                child_module = self.parse_module_from_dict(child_data, None)  # parent 会在 Module 初始化后设置
                children.append(child_module)
        
        # 创建模块对象
        module = Module(
            name=data.get("name", ""),
            module_name=data.get("module_name", ""),
            vertex=data.get("vertex", []),
            direction=data.get("direction", 0),
            color=data.get("color", "#000000"),
            children=children,
            parent=parent
        )
        
        return module


    def load_place_db(self, json_file_path: str) -> Module:
        """
        从 JSON 文件加载模块数据库
        
        Args:
            json_file_path: JSON 文件路径
            
        Returns:
            Module: 根模块对象（通常是 TOP 模块）
            
        Raises:
            FileNotFoundError: 如果文件不存在
            json.JSONDecodeError: 如果 JSON 格式错误
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.parse_module_from_dict(data)


    def load_place_db_from_string(self, json_string: str) -> Module:
        """
        从 JSON 字符串加载模块数据库
        
        Args:
            json_string: JSON 格式的字符串
            
        Returns:
            Module: 根模块对象（通常是 TOP 模块）
            
        Raises:
            json.JSONDecodeError: 如果 JSON 格式错误
        """
        data = json.loads(json_string)
        return self.parse_module_from_dict(data)

    def collect_all_modules(self, module: Module, modules_list: Optional[List[Module]] = None) -> List[Module]:
        """
        递归收集所有模块（包括子模块）
        
        Args:
            module: 根模块
            modules_list: 模块列表（用于递归）
            
        Returns:
            List[Module]: 所有模块的列表
        """
        if modules_list is None:
            modules_list = []
        
        self.all_module_dict[module.name] = module
        modules_list.append(module)
        
        for child in module.children:
            self.collect_all_modules(child, modules_list)
        
        return modules_list

    def load_pingroup_json(self, json_file_path: str) -> List[Net]:
        """
        从 pingroup JSON 文件加载网表信息
        
        Args:
            json_file_path: pingroup JSON 文件路径
            
        Returns:
            List[Net]: 所有网表的列表
            
        Raises:
            FileNotFoundError: 如果文件不存在
            json.JSONDecodeError: 如果 JSON 格式错误
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nets = []
        # data 是一个数组，每个元素是一个网表（也是一个数组，包含多个 Pin）
        for net_data in data:
            pins = []
            for pin_data in net_data:
                pin = Pin(
                    parent_inst=pin_data.get("parent_inst", ""),
                    parent_module=pin_data.get("parent_module", ""),
                    pingroup_name=pin_data.get("pingroup_name", ""),
                    scope=pin_data.get("scope", []),
                    successors=pin_data.get("successors", []),
                    width=pin_data.get("width", 0.0)
                )
                self.all_module_dict[pin.parent_inst].pin_list.append(pin)
                pins.append(pin)
                self.total_pin_count += 1
            net = Net(pins)
            nets.append(net)
        
        return nets
    
    def decompose_multi_fanout_nets(self) -> None:
        """
        分解多扇出 nets：
        1. 将多扇出 pin 拆分为多个模拟 Pin（每个模拟 Pin 单扇出）
        2. 根据连接关系将原 net 拆分成多个独立的子 net
        
        算法原理：
        步骤1 - Pin 分解：
        - 迭代处理，直到没有多扇出 pin
        - 为每个多扇出 pin 创建模拟 Pin
        - 更新连接关系
        
        步骤2 - Net 拆分：
        - 构建 pin 连接图
        - 使用 DFS/BFS 找出所有连通分量
        - 每个连通分量成为一个新的 net
        
        模拟 Pin 命名规则："{original_pingroup_name}#DUMMY#{index}#{original_parent_module}"
        这样在 to_floorplan_ro 中可以解析出原始 pingroup_name 和 parent_module
        
        注意：此方法会修改 self.nets_list，应在 to_floorplan_ro 之前调用
        """
        if not self.nets_list:
            return
        
        DUMMY_SEPARATOR = "#DUMMY#"
        total_dummy_count = 0
        
        # 步骤1: 分解多扇出 pin
        for net in self.nets_list:
            iteration = 0
            max_iterations = 100  # 防止无限循环
            
            while iteration < max_iterations:
                # 找到当前所有多扇出 pin
                multi_fanout_pins = []
                for pin in net.pins:
                    if len(pin.successors) > 1:
                        multi_fanout_pins.append(pin)
                
                if not multi_fanout_pins:
                    break  # 没有多扇出 pin，处理完成
                
                # 为每个多扇出 pin 创建模拟 Pin
                for src_pin in multi_fanout_pins:
                    successors = src_pin.successors.copy()
                    
                    # 创建模拟 Pin 列表
                    dummy_pins = []
                    for i, succ_name in enumerate(successors):
                        dummy_pingroup_name = f"{src_pin.pingroup_name}{DUMMY_SEPARATOR}{i}{DUMMY_SEPARATOR}{src_pin.parent_module}"
                        
                        dummy_pin = Pin(
                            parent_inst=src_pin.parent_inst,
                            parent_module=src_pin.parent_module,
                            pingroup_name=dummy_pingroup_name,
                            scope=list(src_pin.scope),
                            successors=[succ_name],  # 单扇出
                            width=src_pin.width
                        )
                        dummy_pin.x = src_pin.x
                        dummy_pin.y = src_pin.y
                        
                        dummy_pins.append(dummy_pin)
                    
                    # 将模拟 pins 添加到 net
                    net.pins.extend(dummy_pins)
                    total_dummy_count += len(dummy_pins)
                    
                    # 清空原 pin 的 successors
                    src_pin.successors = []
                    
                    # 更新 net 中其他 pin 的 successors
                    src_full_name = src_pin.get_full_name()
                    for pin in net.pins:
                        if pin is src_pin:
                            continue
                        updated_successors = []
                        for succ in pin.successors:
                            if succ == src_full_name:
                                for dummy in dummy_pins:
                                    updated_successors.append(dummy.get_full_name())
                            else:
                                updated_successors.append(succ)
                        pin.successors = updated_successors
                
                iteration += 1
            
            if iteration >= max_iterations:
                print(f"警告: Net 分解达到最大迭代次数，可能存在循环")
        
        # 步骤2: 根据连接关系拆分 net
        new_nets = []
        for net in self.nets_list:
            # 清除用于占位的原始多扇出 pin（successors 已清空）
            active_pins = [pin for pin in net.pins if len(pin.successors) > 0 or
                          any(pin.get_full_name() in p.successors for p in net.pins) or
                          net.pins.count(pin) == 1]  # 保留单扇出 pin 和被其他 pin 连接的 pin
            
            # 如果所有 pin 都被清除，跳过
            if not active_pins:
                continue
            
            # 构建 pin 全名到 pin 对象的映射
            pin_name_to_pin = {}
            for pin in active_pins:
                pin_name_to_pin[pin.get_full_name()] = pin
            
            # 构建无向连接图（邻接表）
            adjacency = {pin: set() for pin in active_pins}
            for pin in active_pins:
                for succ_name in pin.successors:
                    if succ_name in pin_name_to_pin:
                        succ_pin = pin_name_to_pin[succ_name]
                        # 双向连接
                        adjacency[pin].add(succ_pin)
                        adjacency[succ_pin].add(pin)
            
            # 使用 BFS 找出所有连通分量
            visited = set()
            components = []
            
            for pin in active_pins:
                if pin in visited:
                    continue
                
                # BFS 找到连通分量
                component = []
                queue = [pin]
                visited.add(pin)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if component:
                    components.append(component)
            
            # 为每个连通分量创建新的 net
            for component in components:
                # 只包含 successors 指向组件内部的 pin
                component_pins = []
                component_pin_names = {p.get_full_name() for p in component}
                
                for pin in component:
                    # 过滤 successors，只保留指向组件内部的
                    filtered_pin = Pin(
                        parent_inst=pin.parent_inst,
                        parent_module=pin.parent_module,
                        pingroup_name=pin.pingroup_name,
                        scope=list(pin.scope),
                        successors=[s for s in pin.successors if s in component_pin_names],
                        width=pin.width
                    )
                    filtered_pin.x = pin.x
                    filtered_pin.y = pin.y
                    component_pins.append(filtered_pin)
                
                if component_pins:
                    new_net = Net(component_pins)
                    new_nets.append(new_net)
        
        # 更新 nets_list
        old_count = len(self.nets_list)
        self.nets_list = new_nets
        
        print(f"Net 分解完成: 创建了 {total_dummy_count} 个模拟 Pin, "
              f"将 {old_count} 个 nets 拆分为 {len(self.nets_list)} 个 nets")

    def visualize_modules(
        self,
        figsize: tuple = (16, 12),
        show_labels: bool = False,
        alpha: float = 0.7,
        edge_color: str = 'black',
        edge_width: float = 0.5,
        dpi: int = 500,
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化所有模块的布局情况（包括所有子模块）
        
        Args:
            figsize: 图像大小，默认 (16, 12)
            show_labels: 是否显示模块名称标签，默认 False
            alpha: 填充透明度，默认 0.7
            edge_color: 边框颜色，默认 'black'
            edge_width: 边框宽度，默认 0.5
            dpi: 图像分辨率，默认 100
            save_path: 保存路径，如果为 None 则只显示不保存
        """
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 按层级分组模块，先绘制层级浅的（父模块），再绘制层级深的（子模块）
        # 这样可以确保子模块显示在父模块之上，覆盖父模块的颜色
        modules_by_level = {}
        for module in self.all_modules_list:
            level = module.get_hierarchy_level()
            if level not in modules_by_level:
                modules_by_level[level] = []
            modules_by_level[level].append(module)
        # print(f"Modules by level 2: {len(modules_by_level[2])}")
        # 从最浅层级开始绘制（先绘制父模块，后绘制子模块）
        max_level = max(modules_by_level.keys()) if modules_by_level else 0
        
        # 修改：从浅层到深层绘制，确保子模块覆盖父模块
        for level in range(0, max_level + 1):
            if level not in modules_by_level:
                continue
                
            for module in modules_by_level[level]:
                if not module.vertex or len(module.vertex) < 3:
                    continue
                
                # 提取顶点坐标
                vertices = [[v[0], v[1]] for v in module.vertex]
                
                # 对于顶层模块（level 0），不填充颜色，只绘制边框（可选）
                # 如果不需要显示顶层轮廓，可以注释掉这部分
                if level == 0:
                    # 顶层模块：无填充，只显示边框（可选，如果不需要可以跳过）
                    # 如果需要完全隐藏顶层模块，可以使用 continue 跳过
                    polygon = patches.Polygon(
                        vertices,
                        closed=True,
                        facecolor='none',  # 无填充
                        edgecolor=edge_color,  # 使用浅色边框
                        linewidth=edge_width*(max_level - level + 0.5),  # 使用正常线宽
                        alpha=0.1  # 降低透明度，使其不那么明显
                    )
                else:
                    # 子模块：正常填充颜色，会覆盖父模块
                    polygon = patches.Polygon(
                        vertices,
                        closed=True,
                        facecolor=module.color,
                        edgecolor=edge_color,
                        linewidth=edge_width*(max_level - level + 0.5),
                        alpha=alpha
                    )
                
                # 添加到坐标轴
                ax.add_patch(polygon)
                
                # 可选：添加标签
                if show_labels:
                    # 计算多边形的中心点
                    x_coords = [v[0] for v in vertices]
                    y_coords = [v[1] for v in vertices]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    
                    # 添加文本标签
                    ax.text(
                        center_x, center_y,
                        module.module_name,
                        ha='center',
                        va='center',
                        fontsize=6,
                        color='black',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                    )
        
        # 设置坐标轴
        bbox = self.root_module.get_bounding_box()
        margin = 10  # 边距
        ax.set_xlim(bbox["min_x"] - margin, bbox["max_x"] + margin)
        ax.set_ylim(bbox["min_y"] - margin, bbox["max_y"] + margin)
        
        # 设置标题和标签
        ax.set_title("Visualization", fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # 设置等比例
        ax.set_aspect('equal', adjustable='box')
        
        # # 网格（可选）
        # ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        # plt.tight_layout()
        # plt.show()


    def visualize_modules_by_level(
        self,
        root_module: Module,
        max_level: Optional[int] = None,
        figsize: tuple = (16, 12),
        show_labels: bool = False,
        alpha: float = 0.7,
        edge_color: str = 'black',
        edge_width: float = 0.5,
        dpi: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """
        按层级可视化模块（只显示指定层级及以下的模块）
        
        Args:
            root_module: 根模块对象
            max_level: 最大显示层级，None 表示显示所有层级
            figsize: 图像大小，默认 (16, 12)
            show_labels: 是否显示模块名称标签，默认 False
            alpha: 填充透明度，默认 0.7
            edge_color: 边框颜色，默认 'black'
            edge_width: 边框宽度，默认 0.5
            dpi: 图像分辨率，默认 100
            save_path: 保存路径，如果为 None 则只显示不保存
        """
        # 收集所有模块
        all_modules = self.collect_all_modules(root_module)
        
        # 过滤模块（只保留指定层级及以下的）
        if max_level is not None:
            all_modules = [m for m in all_modules if m.get_hierarchy_level() <= max_level]
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 按层级分组模块
        modules_by_level = {}
        for module in all_modules:
            level = module.get_hierarchy_level()
            if level not in modules_by_level:
                modules_by_level[level] = []
            modules_by_level[level].append(module)
        
        # 从最深层级开始绘制
        if modules_by_level:
            max_display_level = max(modules_by_level.keys())
            for level in range(max_display_level, -1, -1):
                if level not in modules_by_level:
                    continue
                    
                for module in modules_by_level[level]:
                    if not module.vertex or len(module.vertex) < 3:
                        continue
                    
                    # 提取顶点坐标
                    vertices = [[v[0], v[1]] for v in module.vertex]
                    
                    # 创建多边形补丁
                    polygon = patches.Polygon(
                        vertices,
                        closed=True,
                        facecolor=module.color,
                        edgecolor=edge_color,
                        linewidth=edge_width,
                        alpha=alpha
                    )
                    
                    # 添加到坐标轴
                    ax.add_patch(polygon)
                    
                    # 可选：添加标签
                    if show_labels:
                        x_coords = [v[0] for v in vertices]
                        y_coords = [v[1] for v in vertices]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        ax.text(
                            center_x, center_y,
                            module.module_name,
                            ha='center',
                            va='center',
                            fontsize=6,
                            color='black',
                            weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                        )
        
        # 设置坐标轴
        bbox = root_module.get_bounding_box()
        margin = 100
        ax.set_xlim(bbox["min_x"] - margin, bbox["max_x"] + margin)
        ax.set_ylim(bbox["min_y"] - margin, bbox["max_y"] + margin)
        
        # 设置标题和标签
        level_info = f" (层级 ≤ {max_level})" if max_level is not None else ""
        ax.set_title(f'模块布局可视化{level_info} (共 {len(all_modules)} 个模块)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 设置等比例
        ax.set_aspect('equal', adjustable='box')
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()

    def convert_to_floorplan_ro(self) -> 'FloorPlanRO':
        """
        将PlaceDB转换为FloorPlanRO格式
        这是为data_model.py中的FloorPlanRO提供的转换接口
        
        主要处理逻辑：
        1. 识别master blocks（复用模块的母模块）
        2. 生成segments（将模块每条边分割成两段）
        3. 创建Block实例（所有层级的模块，不再只考虑最后一层）
        4. 处理Pin和Net
        5. 识别isomorphic pin groups（同构pin组，要求parent_inst不同且pingroup_name相同）
        6. 处理旋转和镜像信息
        7. 设置Block的parent_id和childeren_ids
        """
        try:
            from .data_model import (
                Segment, MasterBlock, Block, Pin, Net,
                IsomorphicPinGroup, FloorPlanRO
            )
        except ImportError:
            # 如果相对导入失败，尝试绝对导入
            from data_model import (
                Segment, MasterBlock, Block, Pin, Net,
                IsomorphicPinGroup, FloorPlanRO
            )
        
        # 0. 首先分解多扇出 nets
        self.decompose_multi_fanout_nets()
        
        # 1. 识别master blocks和生成segments
        master_blocks_dict = {}  # module_name -> master_block_info
        segments = []
        segment_id_counter = 0
        
        # 2. 识别同构pin组（key: pingroup_name, value: {group_id, pin_ids, master_block_id, parent_insts}
        isomorphic_groups = {}  # pingroup_name -> group_info
        
        # 3. 处理所有模块（不再只处理叶子模块），生成segments和master blocks
        for module in self.all_modules_list:
            module_name = module.module_name
            
            # 如果是新的master block
            if module_name not in master_blocks_dict:
                # 生成segments（将每条边分割成两段）
                module_segments = []
                
                # 将模块恢复到rotation=0的原始形状
                original_vertex = self._restore_module_to_original_orientation(module)
                n = len(original_vertex)
                
                for i in range(n):
                    x1, y1 = original_vertex[i]
                    x2, y2 = original_vertex[(i + 1) % n]  # 下一点，形成闭合边
                    
                    # 将边分割成两段
                    mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    
                    # 第一段
                    seg1 = Segment(
                        id=segment_id_counter,
                        edge_id=i,
                        block_id=len(master_blocks_dict),  # 临时使用master block索引
                        x1=x1, y1=y1,
                        x2=mid_x, y2=mid_y,
                        length=((mid_x - x1)**2 + (mid_y - y1)**2)**0.5,
                        max_capacity=10.0,  # 默认容量
                        direction=0 if abs(x2 - x1) > abs(y2 - y1) else 1
                    )
                    segments.append(seg1)
                    module_segments.append(segment_id_counter)
                    segment_id_counter += 1
                    
                    # 第二段
                    seg2 = Segment(
                        id=segment_id_counter,
                        edge_id=i,
                        block_id=len(master_blocks_dict),  # 临时使用master block索引
                        x1=mid_x, y1=mid_y,
                        x2=x2, y2=y2,
                        length=((x2 - mid_x)**2 + (y2 - mid_y)**2)**0.5,
                        max_capacity=((x2 - mid_x)**2 + (y2 - mid_y)**2)**0.5,  # 默认容量
                        direction=0 if abs(x2 - x1) > abs(y2 - y1) else 1
                    )
                    segments.append(seg2)
                    module_segments.append(segment_id_counter)
                    segment_id_counter += 1
                
                # 计算原始形状中x+y坐标最大的点作为参考点
                max_sum_idx = 0
                max_sum = original_vertex[0][0] + original_vertex[0][1]
                for i, (vx, vy) in enumerate(original_vertex):
                    if vx + vy > max_sum:
                        max_sum = vx + vy
                        max_sum_idx = i
                
                # 判断母模块建立时是否有镜像
                # 根据模块direction判断
                direction = int(module.direction)
                is_master_mirrored = 1 if direction in [4, 5, 6, 7] else 0  # MY, MX, MX90, MY90
                
                master_blocks_dict[module_name] = {
                    'id': len(master_blocks_dict),
                    'name': module_name,
                    'segments': tuple(module_segments),
                    'position': (original_vertex[0][0], original_vertex[0][1]),  # 使用原始形状的第一个点坐标
                    'reference_point_idx': max_sum_idx,  # x+y最大点的索引
                    'is_mirrored': is_master_mirrored    # 母模块建立时的镜像状态
                }
        
        # 4. 创建MasterBlock对象
        master_blocks = []
        master_blocks_by_id = {}  # 用于通过master_id快速查找
        for info in master_blocks_dict.values():
            master_block = MasterBlock(
                id=info['id'],
                name=info['name'],
                segments_ids=info['segments'],
                position=info['position']
            )
            master_blocks.append(master_block)
            master_blocks_by_id[info['id']] = info  # 保存完整信息
        
        # 5. 创建Block实例（所有层级的模块）
        blocks = []
        block_id_counter = 0
        block_name_to_id = {}  # 提前创建映射
        
        for module in self.all_modules_list:
            module_name = module.module_name
            master_id = master_blocks_dict[module_name]['id']
            
            # 计算模块位置（质心）
            point_num = len(module.vertex)
            cx: float = 0.0
            cy: float = 0.0
            for i, line in enumerate(module.vertex):
                cx += line[0]
                cy += line[1]
            position = (cx / point_num, cy / point_num)
            
            # 处理旋转和镜像 - 使用映射字典替代冗长的if-else
            # 根据CoordRotation枚举，将direction映射到实际旋转角度和镜像类型
            direction = int(module.direction)
            
            # 定义旋转和镜像映射：direction -> (rotation_angle, mirror_axis)
            # mirror_axis: 0=无镜像, 1=Y轴镜像, 2=X轴镜像
            rotation_mirror_map = {
                0: (0, 0),      # ROTATION_R0: 无旋转，无镜像
                1: (180, 0),    # ROTATION_R180: 180度旋转，无镜像
                2: (90, 0),     # ROTATION_R90: 90度旋转，无镜像
                3: (270, 0),    # ROTATION_R270: 270度旋转，无镜像
                4: (0, 1),      # ROTATION_MY: 无旋转，Y轴镜像
                5: (0, 2),      # ROTATION_MX: 无旋转，X轴镜像
                6: (90, 2),     # ROTATION_MX90: 90度旋转，X轴镜像
                7: (90, 1),     # ROTATION_MY90: 90度旋转，Y轴镜像
                -1: (0, 0)      # UNDEFINE: 默认为无旋转无镜像
            }
            
            # 获取旋转角度和镜像类型
            rotation, mirror_axis = rotation_mirror_map.get(direction, (0, 0))
            is_mirrored = 1 if mirror_axis > 0 else 0  # 简化镜像标记：1表示有镜像，0表示无镜像
            
            block = Block(
                id=block_id_counter,
                name=module.name,
                master_id=master_id,
                parent_id=None,  # 稍后设置
                childeren_ids=None,  # 稍后设置
                position=position,
                rotation=rotation,
                is_mirrored=is_mirrored,
                vertex=module.vertex
            )
            blocks.append(block)
            block_name_to_id[module.name] = block.id  # 立即添加到映射
            block_id_counter += 1
        
        # 6. 设置Block的parent_id和childeren_ids
        for module in self.all_modules_list:
            if module.parent is not None:
                block_id = block_name_to_id[module.name]
                parent_id = block_name_to_id[module.parent.name]
                
                # 设置parent_id
                blocks[block_id] = Block(
                    id=blocks[block_id].id,
                    name=blocks[block_id].name,
                    master_id=blocks[block_id].master_id,
                    parent_id=parent_id,
                    childeren_ids=blocks[block_id].childeren_ids,
                    position=blocks[block_id].position,
                    rotation=blocks[block_id].rotation,
                    is_mirrored=blocks[block_id].is_mirrored,
                    vertex=blocks[block_id].vertex
                )
                
                # 设置parent的childeren_ids
                parent_block = blocks[parent_id]
                if parent_block.childeren_ids is None:
                    childeren_ids = [block_id]
                else:
                    childeren_ids = list(parent_block.childeren_ids) + [block_id]
                
                blocks[parent_id] = Block(
                    id=parent_block.id,
                    name=parent_block.name,
                    master_id=parent_block.master_id,
                    parent_id=parent_block.parent_id,
                    childeren_ids=tuple(childeren_ids),
                    position=parent_block.position,
                    rotation=parent_block.rotation,
                    is_mirrored=parent_block.is_mirrored,
                    vertex=parent_block.vertex
                )
        
        # 7. 创建实例级别的segment_inst - 基于当前segments和block变换
        segment_insts = []
        segment_inst_id_counter = 0
        
        # 为每个block实例创建其对应的segment_inst
        for block in blocks:
            # 获取母模块的详细信息（使用master_id查找）
            master_block_info = master_blocks_by_id.get(block.master_id)
            if not master_block_info:
                continue
            
            # 获取当前block的segments（从母模块的segments_ids）
            master_block_obj = master_blocks[block.master_id]
            block_segments = []
            for seg_id in master_block_obj.segments_ids:
                if seg_id < len(segments):
                    block_segments.append(segments[seg_id])
            
            if not block_segments:
                continue
            
            # 获取母模块的参考点信息和镜像状态
            master_ref_idx = master_block_info.get('reference_point_idx', 0)
            master_is_mirrored = master_block_info.get('is_mirrored', 0)
            
            # 将block的vertex坐标应用到segments上，得到真实的实例坐标
            transformed_segments = self._transform_segments_to_instance_coordinates(
                block, master_ref_idx, master_is_mirrored
            )
            
            # 为每个变换后的segment创建SegmentInst
            for i, (transformed_seg, original_seg) in enumerate(zip(transformed_segments, block_segments)):
                segment_inst = SegmentInst(
                    id=segment_inst_id_counter,
                    segment_id=original_seg.id,
                    block_id=block.id,
                    x1=transformed_seg[0],
                    y1=transformed_seg[1],
                    x2=transformed_seg[2],
                    y2=transformed_seg[3],
                    length=math.sqrt((transformed_seg[2] - transformed_seg[0])**2 + (transformed_seg[3] - transformed_seg[1])**2),
                    direction=0 if abs(transformed_seg[2] - transformed_seg[0]) > abs(transformed_seg[3] - transformed_seg[1]) else 1
                )
                segment_insts.append(segment_inst)
                segment_inst_id_counter += 1
        
        # Segment实例创建完成
        
        # 8. 处理Pin和Net
        pins = []
        nets = []
        pin_id_counter = 0
        net_id_counter = 0
        
        for net in self.nets_list:
            net_pin_ids = []
            
            for pin in net.pins:
                # 创建Pin对象
                pin_name = f"{pin.parent_inst}.{pin.pingroup_name}"
                
                # 查找所属block
                block_id = block_name_to_id.get(pin.parent_inst, -1)
                if block_id == -1:
                    continue  # 跳过无法找到block的pin
            
                pin_obj = Pin(
                    id=pin_id_counter,
                    name=pin_name,
                    block_id=block_id,
                    width=pin.width,
                    net_id=net_id_counter,
                    isomorphic_group_id=None  # 稍后设置
                )
                pins.append(pin_obj)
                
                # 识别同构组：要求pingroup_name相同且parent_module相同
                # 对于模拟 Pin（多扇出拆分产生的），解析原始 pingroup_name
                DUMMY_SEPARATOR = "#DUMMY#"
                if DUMMY_SEPARATOR in pin.pingroup_name:
                    # 格式: "original_name#DUMMY#index#original_module"
                    parts = pin.pingroup_name.split(DUMMY_SEPARATOR)
                    original_pingroup_name = parts[0]
                    original_parent_module = parts[2] if len(parts) > 2 else pin.parent_module
                    group_key = (original_pingroup_name, original_parent_module)
                else:
                    group_key = (pin.pingroup_name, pin.parent_module)
                if group_key not in isomorphic_groups:
                    isomorphic_groups[group_key] = {
                        'group_id': len(isomorphic_groups),
                        'pin_ids': [],
                        'master_block_id': None
                    }
                
                group_info = isomorphic_groups[group_key]
                group_info['pin_ids'].append(pin_obj.id)
                
                # 确保同构组内的pin属于相同的master block
                block = blocks[block_id]
                if group_info['master_block_id'] is None:
                    group_info['master_block_id'] = block.master_id
                elif group_info['master_block_id'] != block.master_id:
                    print(f"警告: Pin {pin_name} 的同构组master block不一致")
                
                net_pin_ids.append(pin_id_counter)
                pin_id_counter += 1
            
            # 创建Net对象
            if net_pin_ids:  # 只创建有pin的net
                net_obj = Net(
                    id=net_id_counter,
                    name=f"net_{net_id_counter}",
                    pin_ids=tuple(net_pin_ids)
                )
                nets.append(net_obj)
                net_id_counter += 1
        
        # 8. 创建IsomorphicPinGroup对象（只包含多个pin的组）
        isomorphic_pin_groups = []
        pin_to_group = {}
        
        for group_key, group_info in isomorphic_groups.items():
            # 只有多个pin才构成同构组
            # if len(group_info['pin_ids']) > 1:
                group = IsomorphicPinGroup(
                    group_id=group_info['group_id'],
                    pin_ids=tuple(group_info['pin_ids']),
                    master_block_id=group_info['master_block_id']
                )
                isomorphic_pin_groups.append(group)
                
                # 建立pin到group的映射
                for pin_id in group_info['pin_ids']:
                    pin_to_group[pin_id] = group.group_id
        
        # 9. 更新Pin的同构组ID
        updated_pins = []
        for pin in pins:
            if pin.id in pin_to_group:
                updated_pin = Pin(
                    id=pin.id,
                    name=pin.name,
                    block_id=pin.block_id,
                    width=pin.width,
                    net_id=pin.net_id,
                    isomorphic_group_id=pin_to_group[pin.id]
                )
                updated_pins.append(updated_pin)
            else:
                updated_pins.append(pin)
        
        # 10. 计算芯片尺寸
        if segments:
            chip_length = max(
                max(seg.x1 for seg in segments),
                max(seg.x2 for seg in segments),
                max(seg.y1 for seg in segments),
                max(seg.y2 for seg in segments)
            )
        else:
            chip_length = 1000.0
        
        # 11. 创建FloorPlanRO对象
        floorplan_ro = FloorPlanRO(
            segments=segments,
            segment_insts=segment_insts,
            master_blocks=master_blocks,
            blocks=blocks,
            pins=updated_pins,
            nets=nets,
            isomorphic_pin_groups=isomorphic_pin_groups,
            chip_length=chip_length
        )
        
        return floorplan_ro
    
    def _transform_segments_to_instance_coordinates(self, block: Block,
                                                     master_ref_idx: int = 0, master_is_mirrored: int = 0) -> List[Tuple[float, float, float, float]]:
        """将segment坐标转换到实例坐标系 - 基于母模块和实例block的点对应关系"""
        transformed_segments = []
        
        if not block.vertex or len(block.vertex) < 2:
            return transformed_segments
        
        # 步骤1: 将block还原为rotation=0的方向
        original_vertices = self._restore_block_to_original_orientation(block)
        
        if not original_vertices:
            return transformed_segments
        
        # 步骤2: 找到当前实例还原后x+y坐标最大的点作为参考点
        inst_ref_idx = 0
        max_sum = original_vertices[0][0] + original_vertices[0][1]
        for i, (vx, vy) in enumerate(original_vertices):
            if vx + vy > max_sum:
                max_sum = vx + vy
                inst_ref_idx = i
        
        # 步骤3: 确定是否需要反转顶点顺序
        # 镜像操作会反转顶点顺序（逆时针变顺时针）
        # 需要比较母模块建立时的镜像状态和当前block的镜像状态
        need_reverse = (block.is_mirrored != master_is_mirrored)
        
        # 步骤4: 建立点映射: 母模块顶点索引 -> 实例顶点索引
        # 将母模块的参考点与实例的参考点对齐
        n = len(original_vertices)
        point_mapping = {}
        
        for i in range(n):
            # 计算相对于母模块参考点的偏移
            master_offset = (i - master_ref_idx) % n
            
            if need_reverse:
                # 需要反转顺序
                inst_idx = (inst_ref_idx - master_offset) % n
            else:
                # 保持顺序
                inst_idx = (inst_ref_idx + master_offset) % n
            
            point_mapping[i] = inst_idx
        
        # 步骤5: 像母模块一样直接创建实例segments
        # 遍历每条边，将边分成两段（与母模块方式相同）
        for i in range(n):
            # 获取母模块边的起点和终点索引
            master_start_idx = i
            master_end_idx = (i + 1) % n
            
            # 获取对应的实例顶点索引
            inst_start_idx = point_mapping[master_start_idx]
            inst_end_idx = point_mapping[master_end_idx]
            
            # 获取实例边的实际坐标
            vx1, vy1 = block.vertex[inst_start_idx]
            vx2, vy2 = block.vertex[inst_end_idx]
            
            # 将边分成两段（与母模块的生成方式一致）
            mid_x, mid_y = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
            
            # 第一段
            transformed_segments.append((vx1, vy1, mid_x, mid_y))
            # 第二段
            transformed_segments.append((mid_x, mid_y, vx2, vy2))
        
        return transformed_segments
    
    def _restore_module_to_original_orientation(self, module) -> List[Tuple[float, float]]:
        """将模块恢复到rotation=0的原始形状"""
        if not module.vertex or len(module.vertex) < 2:
            return []
        
        original_vertices = []
        
        # 获取当前顶点
        current_vertices = module.vertex
        
        # 根据旋转方向进行逆向变换
        direction = int(module.direction)
        
        # 定义旋转和镜像映射：direction -> (rotation_angle, mirror_axis)
        # mirror_axis: 0=无镜像, 1=Y轴镜像, 2=X轴镜像
        rotation_mirror_map = {
            0: (0, 0),      # ROTATION_R0: 无旋转，无镜像
            1: (180, 0),    # ROTATION_R180: 180度旋转，无镜像
            2: (90, 0),     # ROTATION_R90: 90度旋转，无镜像
            3: (270, 0),    # ROTATION_R270: 270度旋转，无镜像
            4: (0, 1),      # ROTATION_MY: 无旋转，Y轴镜像
            5: (0, 2),      # ROTATION_MX: 无旋转，X轴镜像
            6: (90, 2),     # ROTATION_MX90: 90度旋转，X轴镜像
            7: (90, 1),     # ROTATION_MY90: 90度旋转，Y轴镜像
            -1: (0, 0)      # UNDEFINE: 默认为无旋转无镜像
        }
        
        # 获取旋转角度和镜像类型
        rotation, mirror_axis = rotation_mirror_map.get(direction, (0, 0))
        
        for vx, vy in current_vertices:
            # 先处理镜像（如果有）
            if mirror_axis == 1:  # Y轴镜像
                vx = -vx
            elif mirror_axis == 2:  # X轴镜像
                vy = -vy
            
            # 处理旋转的逆向变换
            if rotation == 90:
                # 90度旋转的逆向是270度旋转
                new_x = vy
                new_y = -vx
            elif rotation == 180:
                # 180度旋转的逆向是180度旋转
                new_x = -vx
                new_y = -vy
            elif rotation == 270:
                # 270度旋转的逆向是90度旋转
                new_x = -vy
                new_y = vx
            else:  # rotation == 0 or other
                new_x = vx
                new_y = vy
            
            original_vertices.append((new_x, new_y))
        
        return original_vertices
    
    def _restore_block_to_original_orientation(self, block: Block) -> List[Tuple[float, float]]:
        """将block还原为rotation=0的原始形状"""
        if not block.vertex or len(block.vertex) < 2:
            return []
        
        original_vertices = []
        
        # 获取当前顶点
        current_vertices = block.vertex
        
        # 根据旋转角度进行逆向变换
        rotation = block.rotation
        is_mirrored = block.is_mirrored
        
        for vx, vy in current_vertices:
            # 先处理镜像（如果有）
            if is_mirrored == 1:  # 有镜像
                # 根据block的创建逻辑，镜像主要是Y轴镜像
                vx = -vx
            
            # 处理旋转的逆向变换
            if rotation == 90:
                # 90度旋转的逆向是270度旋转
                new_x = vy
                new_y = -vx
            elif rotation == 180:
                # 180度旋转的逆向是180度旋转
                new_x = -vx
                new_y = -vy
            elif rotation == 270:
                # 270度旋转的逆向是90度旋转
                new_x = -vy
                new_y = vx
            else:  # rotation == 0 or other
                new_x = vx
                new_y = vy
            
            original_vertices.append((new_x, new_y))
        
        return original_vertices
    
    def _create_point_mapping_between_original_and_instance(self, original_vertices: List[Tuple[float, float]],
                                                           instance_vertices: List[Tuple[float, float]],
                                                           original_max_idx: int, instance_max_idx: int) -> Dict[int, int]:
        """建立原始顶点和实例顶点之间的映射关系"""
        mapping = {}
        
        if len(original_vertices) != len(instance_vertices):
            return mapping
        
        n = len(original_vertices)
        
        # 找到最佳的起始点偏移（考虑旋转和镜像对点顺序的影响）
        best_offset = 0
        best_score = -1
        
        for offset in range(n):
            score = 0
            for i in range(n):
                orig_idx = (original_max_idx + i) % n
                inst_idx = (instance_max_idx + i + offset) % n
                
                orig_x, orig_y = original_vertices[orig_idx]
                inst_x, inst_y = instance_vertices[inst_idx]
                
                # 计算相似度
                distance = math.sqrt((orig_x - inst_x)**2 + (orig_y - inst_y)**2)
                similarity = 1.0 / (1.0 + distance)  # 距离越小，相似度越高
                score += similarity
            
            if score > best_score:
                best_score = score
                best_offset = offset
        
        # 建立映射
        for i in range(n):
            orig_idx = (original_max_idx + i) % n
            inst_idx = (instance_max_idx + i + best_offset) % n
            mapping[orig_idx] = inst_idx
        
        return mapping
    
    def _align_segment_direction(self, original_segment: Segment, inst_start: Tuple[float, float], inst_end: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """对齐segment方向，确保与原始segment方向一致"""
        orig_dx = original_segment.x2 - original_segment.x1
        orig_dy = original_segment.y2 - original_segment.y1
        
        inst_dx = inst_end[0] - inst_start[0]
        inst_dy = inst_end[1] - inst_start[1]
        
        # 检查方向是否一致（考虑180度旋转的情况）
        direction_consistent = (orig_dx * inst_dx + orig_dy * inst_dy) >= 0
        
        if direction_consistent:
            # 方向一致，保持原顺序
            return (inst_start[0], inst_start[1], inst_end[0], inst_end[1])
        else:
            # 方向相反，需要反转
            return (inst_end[0], inst_end[1], inst_start[0], inst_start[1])
    
    def _find_segment_edge_in_vertices(self, vertices: List[Tuple[float, float]], seg_start: Tuple[float, float], seg_end: Tuple[float, float]) -> int:
        """找到segment在vertices中的对应边"""
        if not vertices or len(vertices) < 2:
            return -1
        
        # 计算segment的中点和特征
        seg_cx = (seg_start[0] + seg_end[0]) / 2.0
        seg_cy = (seg_start[1] + seg_end[1]) / 2.0
        seg_length = math.sqrt((seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2)
        
        best_edge_idx = -1
        min_distance = float('inf')
        
        for i in range(len(vertices)):
            vx1, vy1 = vertices[i]
            vx2, vy2 = vertices[(i + 1) % len(vertices)]
            
            # 计算边的中点和长度
            edge_cx = (vx1 + vx2) / 2.0
            edge_cy = (vy1 + vy2) / 2.0
            edge_length = math.sqrt((vx2 - vx1)**2 + (vy2 - vy1)**2)
            
            # 计算距离和长度差异
            distance = math.sqrt((seg_cx - edge_cx)**2 + (seg_cy - edge_cy)**2)
            length_diff = abs(seg_length - edge_length)
            
            # 综合评分
            score = distance + length_diff * 0.1  # 长度差异权重较小
            
            if score < min_distance:
                min_distance = score
                best_edge_idx = i
        
        return best_edge_idx

# 使用示例
if __name__ == "__main__":
    # 示例：加载 block.json 文件

    database = PlaceDB("/root/autodl-tmp/pin_assignment/test_case_from_huawei/block.json", "/root/autodl-tmp/pin_assignment/test_case_from_huawei/pingroup.json")

    print(database)
    database.visualize_modules(save_path="module_layout.png")
    
    print(len(database.nets_list))
    # 转换为FloorPlanRO
    floorplan_ro = database.convert_to_floorplan_ro()
    print(f"\n转换后的FloorPlanRO:")
    print(f"Segments: {floorplan_ro.num_segments}")
    print(f"MasterBlocks: {floorplan_ro.num_master_blocks}")
    print(f"Blocks: {floorplan_ro.num_blocks}")
    print(f"Pins: {floorplan_ro.num_pins}")
    print(f"Nets: {floorplan_ro.num_nets}")
    print(f"IsomorphicGroups: {len([g for g in floorplan_ro._isomorphic_groups.values() if g.size > 1])}")


    # print(f"\n测试查询Pin1 的 block:")
    # print(floorplan_ro.get_block_by_name("TOP.U_T1_0.U_T13_1"))
    # test_net = floorplan_ro.get_net(751)
    # print(f"\n测试查询Net160:")
    # print(test_net.pin_ids)
    # test_pin3 = floorplan_ro.get_pin(33)
    # print(f"\n测试查询Pin3:")
    # print(test_pin3)
    # print(f"\n测试查询pin")
    # test_pin = floorplan_ro.get_pin_by_name("TOP.U_H.U_H4.H4_dst_h3_h4_1_6")
    # print(test_pin)
    print(len(database.nets_list))


    
