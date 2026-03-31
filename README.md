# mcts-with-PPO
下载后先将 src_version v3-1 文件夹解压到原位置

##### 使用方法
###### 1.模型训练
- 直接执行src_version 文件中的 ppo_training.py 文件，模型参数存储在trained_models文件夹下
- 训练前需更改case的路径，源文件中使用绝对路径，在ppo_trainging文件的763行，其他训练参数也在同一个函数中
###### 2.执行模拟
- 直接执行src_version 文件中的 simple_search_with_models.py 文件，输出结果存储在src_version文件夹下
- 参数修改(包括文件路径)在37行的CONFIG中修改，模型文件、数据文件、feedthrough预测器路径均在当中
###### 3.输出结果
- net_results依照test_case中的pin_group文件中的net顺序一一对应，每个assignments是一个net的分配结果，pin顺序也保持一致；seg_coords是当前pin分配的segment位置信息，x1和y1为以及x2和y2分别为该段头尾坐标，midpoint为中点
- 当前输出结果未针对复用模块进行硬约束，并且因为模型结果不好会有大量的复用pin位置可分配位置冲突，这一点在模型优化后仍是不可避免的，输出的pin位置仍会有部分冲突
