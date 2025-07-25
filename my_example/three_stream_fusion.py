import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

# 三流融合架构 (Three-Stream Fusion Architecture)
# 重新设计为真正的三流输入架构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # ============ 第一流：静态代价空间流 (上方) ============
    # 静态代价空间输入
    to_Conv("static_input", 32, 1, offset="(0,4,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Static CV\\\\Input"),
    
    # 静态流处理
    to_Conv("static_conv1", 32, 16, offset="(2,0,0)", to="(static_input-east)", 
            height=32, depth=32, width=2, caption="Static\\\\Conv1"),
    to_connection("static_input", "static_conv1"),
    
    to_Conv("static_conv2", 28, 32, offset="(2,0,0)", to="(static_conv1-east)", 
            height=28, depth=28, width=3, caption="Static\\\\Conv2"),
    to_connection("static_conv1", "static_conv2"),
    
    # ============ 第二流：动态代价空间流 (中间) ============
    # 动态代价空间输入
    to_Conv("dynamic_input", 32, 1, offset="(0,0,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Dynamic CV\\\\Input"),
    
    # 动态流处理
    to_Conv("dynamic_conv1", 32, 16, offset="(2,0,0)", to="(dynamic_input-east)", 
            height=32, depth=32, width=2, caption="Dynamic\\\\Conv1"),
    to_connection("dynamic_input", "dynamic_conv1"),
    
    to_Conv("dynamic_conv2", 28, 32, offset="(2,0,0)", to="(dynamic_conv1-east)", 
            height=28, depth=28, width=3, caption="Dynamic\\\\Conv2"),
    to_connection("dynamic_conv1", "dynamic_conv2"),
    
    # ============ 第三流：光流输入流 (下方) ============
    # 光流输入
    to_Conv("flow_input", 64, 2, offset="(0,-4,0)", to="(0,0,0)", 
            height=64, depth=64, width=1, caption="Flow\\\\Input"),
    
    # 光流处理
    to_Conv("flow_conv1", 32, 8, offset="(2,0,0)", to="(flow_input-east)", 
            height=32, depth=32, width=2, caption="Flow\\\\Conv1"),
    to_connection("flow_input", "flow_conv1"),
    
    to_Conv("flow_conv2", 28, 16, offset="(2,0,0)", to="(flow_conv1-east)", 
            height=28, depth=28, width=2.5, caption="Flow\\\\Conv2"),
    to_connection("flow_conv1", "flow_conv2"),
    
    # ============ 特征融合阶段 ============
    # 三流特征融合点
    to_Conv("fusion_concat", 32, 80, offset="(3,0,0)", to="(dynamic_conv2-east)", 
            height=32, depth=32, width=5, caption="Three-Stream\\\\Fusion\\\\32+32+16=80"),
    
    # 连接三个流到融合点
    to_connection("static_conv2", "fusion_concat"),
    to_connection("dynamic_conv2", "fusion_concat"),
    to_connection("flow_conv2", "fusion_concat"),
    
    # ============ 掩码生成分支 ============
    # 从融合特征生成掩码
    to_Conv("mask_gen", 24, 1, offset="(0,3,0)", to="(fusion_concat-east)", 
            height=24, depth=24, width=1, caption="Mask\\\\Generation"),
    to_connection("fusion_concat", "mask_gen"),
    
    # ============ 残差融合网络 ============
    # 融合网络第一层
    to_Conv("fusion_conv1", 30, 64, offset="(2,0,0)", to="(fusion_concat-east)", 
            height=30, depth=30, width=4, caption="Fusion\\\\Conv1\\\\80→64"),
    to_connection("fusion_concat", "fusion_conv1"),
    
    # 应用掩码
    to_Conv("masked_feat", 28, 64, offset="(1.5,0,0)", to="(fusion_conv1-east)", 
            height=28, depth=28, width=4, caption="Masked\\\\Features"),
    to_connection("fusion_conv1", "masked_feat"),
    to_connection("mask_gen", "masked_feat"),
    
    # 融合网络第二层
    to_Conv("fusion_conv2", 26, 32, offset="(2,0,0)", to="(masked_feat-east)", 
            height=26, depth=26, width=3, caption="Fusion\\\\Conv2\\\\64→32"),
    to_connection("masked_feat", "fusion_conv2"),
    
    # 最终输出层
    to_Conv("final_output", 24, 1, offset="(2,0,0)", to="(fusion_conv2-east)", 
            height=24, depth=24, width=1, caption="Final\\\\Output\\\\32→1"),
    to_connection("fusion_conv2", "final_output"),
    
    # ============ 跳跃连接 ============
    # 从三个输入流到最终输出的跳跃连接
    to_skip("static_input", "final_output", pos=1.5),
    to_skip("dynamic_input", "final_output", pos=1.3),
    to_skip("flow_input", "final_output", pos=1.1),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
