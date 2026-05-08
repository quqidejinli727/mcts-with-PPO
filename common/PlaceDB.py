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
import torch

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

class Edge:
    """
    Edge 类，表示模块的一条边
    
    属性:
        start_val: 边的起点坐标
        end_val: 边的终点坐标
        fixed_val: 固定的坐标值
        direction: 边的方向, 'horizontal' or 'vertical'
        index: 边在模块中的索引
    """
    
    def __init__(self, start_val: int, end_val: int, fixed_val: int, direction: str, index: int):
        self.start_point = start_val
        self.end_point = end_val
        self.length = abs(start_val - end_val)
        self.direction = direction
        self.fixed_val = fixed_val
        self.pins = []
        self.pin_positions_1d = []
        self.pin_widths = []
        
    
    def add_pin(self, pin: Pin):
        self.pins.append(pin)
        self.pin_positions_1d.append(pin.x)
    
    def add_pin_width_list(self, pin_widths: List[float]):
        self.pin_widths = pin_widths
    
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
        self.edges_list = []
        
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
    
    def get_all_edges(self) -> List[Edge]:
        """
        获取模块的所有边
        
        Returns:
            List[Edge]: 模块的所有边
        """
        num_vertices = len(self.vertex)
        for i in range(num_vertices):
            start_point = tuple(self.vertex[i])
            if i == num_vertices - 1:
                # 最后一条边：从最后一个顶点连接到第一个顶点
                end_point = tuple(self.vertex[0])
            else:
                # 其他边：从当前顶点连接到下一个顶点
                end_point = tuple(self.vertex[i + 1])
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            if abs(dy) < 1e-6:  # 水平边
                direction = 'horizontal'
                edge = Edge(start_point[0], end_point[0], start_point[1], direction, i)
            elif abs(dx) < 1e-6:  # 垂直边
                direction = 'vertical'
                edge = Edge(start_point[1], end_point[1], start_point[0], direction, i)
            else:
                raise ValueError(f"边 {i} 不是水平或垂直边")
            self.edges_list.append(edge)
        
        return self.edges_list
            
            
    
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
        all_modules_list: List[List[Edges]]
            all edges of all modules, sorted by module index
    """
    def __init__(self, block_json_file_path: str, pingroup_json_file_path: str, device: str = "cpu"):
        """
        初始化 PlaceDB 对象
        
        Args:
            block_json_file_path: block.json 文件路径
            pingroup_json_file_path: pingroup.json 文件路径
        """
        # self.root_module = self.load_place_db(block_json_file_path)
        self.all_module_dict = {}
        self.all_edges_list = []
        # self.all_modules_list = self.collect_all_modules(self.root_module)
        self.total_pin_count = 0
        # self.nets_list = self.load_pingroup_json(pingroup_json_file_path)
        # self.device = torch.device("cuda" if params.gpu else "cpu")
        # self.flat_net2pin_map, self.flat_net2pin_start_map, self.pin2net_map, self.net_weights, self.net_mask = self.build_net_data_structures(self.nets_list, device)
        self.flat_net2pin_map = None
        self.flat_net2pin_start_map = None
        self.pin2net_map = None
        self.net_weights = None
        self.net_mask = None
        self.nets_list = None
    
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

    def build_net_data_structures(self, nets, device):
        """
        @brief Build net data structures for wirelength computation
        @param nets list of nets, each net is a list of pin indices
        @param num_pins total number of pins
        @param device torch device
        @return flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_weights, net_mask
        """
        # Build flat_net2pin_map and flat_net2pin_start_map
        flat_net2pin_map = []
        flat_net2pin_start_map = [0]
        
        for net in nets:
            flat_net2pin_map.extend(net)
            flat_net2pin_start_map.append(len(flat_net2pin_map))
        
        flat_net2pin_map = torch.tensor(flat_net2pin_map, dtype=torch.int64, device=device)
        flat_net2pin_start_map = torch.tensor(flat_net2pin_start_map, dtype=torch.int64, device=device)
        
        # Build pin2net_map
        pin2net_map = torch.full((self.total_pin_count,), -1, dtype=torch.int32, device=device)
        for net_id, net in enumerate(nets):
            for pin_id in net:
                pin2net_map[pin_id] = net_id
        
        # Net weights (all ones by default)
        num_nets = len(nets)
        net_weights = torch.ones(num_nets, dtype=torch.float32, device=device)
        
        # Net mask (all True by default, skip single-pin nets)
        net_mask = torch.tensor([len(net) >= 2 for net in nets], dtype=torch.bool, device=device)
        
        return flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_weights, net_mask

    
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
        self.all_edges_list.append(module.get_all_edges())
        
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

# 使用示例
if __name__ == "__main__":
    # 示例：加载 block.json 文件
    
    database = PlaceDB("block.json", "pingroup.json")
    print(database)
    database.visualize_modules(save_path="module_layout.png")

