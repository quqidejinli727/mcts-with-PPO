# HUAWEI
## MCTS
### 使用方法
- 执行 complete_mcts_pipeline.py 文件中的 main 函数
- 传入参数：
  block_file:block 文件路径
  pingroup_file：网表文件路径
  num_simulations 单个net mcts模拟次数
  time_limit: 单个net mcts模拟时间上限
部分参数配置受 multi_net_isomorphic_mcts 中19行 MCTSConfig限制，见complete_mcts_pipeline.py文件224行
文件路径建议使用绝对路径
### 文件输出
在 src 文件同一级目录下产生文件夹mcts_result,其中final_result中存储的是最后输出文件，按segment排列，具体见segment_assignments_README
## Feedthrough Evaluation

本项目用于对布局结果进行评估，输出：

- 全部 nets 的 `Total HPWL`
- 全部可计算 nets 的 `Total Feedthrough`

当前主要入口脚本为 `evaluate.py`。

### 目录结构

```text
feedthrough/
├─ evaluate.py
├─ ftpred_loader.py
├─ PlaceDB.py
├─ build/
│  └─ ftpred.exe
└─ benchmark/
   ├─ input/
   │  ├─ block.json
   │  └─ pingroup.json
   └─ result/
      └─ result.json
```

### 输入文件说明

- `benchmark/input/block.json`：模块几何与层级信息
- `benchmark/input/pingroup.json`：网表/引脚拓扑（原始）
- `benchmark/result/result.json`：引脚最终坐标（`scope=[x,y]`）

`evaluate.py` 会读取 `result.json` 计算 HPWL，并把坐标映射回 `PlaceDB` 后调用 `ftpred` 计算 feedthrough 总和。

### 运行方式

在项目根目录执行：

```powershell
py .\evaluate.py
```

也可以显式传参：

```powershell
py .\evaluate.py .\benchmark\result\result.json .\benchmark\input\block.json .\benchmark\input\pingroup.json .\build\ftpred.exe
```

### 输出示例

程序会输出类似：

- `Total HPWL = ...`
- `Total Feedthrough = ...`
