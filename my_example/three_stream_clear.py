import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

# 清晰的三流融合架构 (Clear Three-Stream Fusion Architecture)
# 参考UNet结构，确保每一流都清晰分离
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # ============ 第一流：静态代价空间流 (上方流) ============
    to_Conv("static_input", 32, 1, offset="(0,4,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Static Input"),
    
    to_Conv("static_proc1", 32, 16, offset="(2,0,0)", to="(static_input-east)", 
            height=32, depth=32, width=2, caption="Static\\\\Process1"),
    to_connection("static_input", "static_proc1"),
    
    to_Conv("static_proc2", 28, 32, offset="(2,0,0)", to="(static_proc1-east)", 
            height=28, depth=28, width=3, caption="Static\\\\Process2"),
    to_connection("static_proc1", "static_proc2"),
    
    # ============ 第二流：动态代价空间流 (中间流) ============
    to_Conv("dynamic_input", 32, 1, offset="(0,0,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Dynamic Input"),
    
    to_Conv("dynamic_proc1", 32, 16, offset="(2,0,0)", to="(dynamic_input-east)", 
            height=32, depth=32, width=2, caption="Dynamic\\\\Process1"),
    to_connection("dynamic_input", "dynamic_proc1"),
    
    to_Conv("dynamic_proc2", 28, 32, offset="(2,0,0)", to="(dynamic_proc1-east)", 
            height=28, depth=28, width=3, caption="Dynamic\\\\Process2"),
    to_connection("dynamic_proc1", "dynamic_proc2"),
    
    # ============ 第三流：光流输入流 (下方流) ============
    to_Conv("flow_input", 64, 2, offset="(0,-4,0)", to="(0,0,0)", 
            height=64, depth=64, width=1, caption="Flow Input"),
    
    to_Conv("flow_proc1", 32, 8, offset="(2,0,0)", to="(flow_input-east)", 
            height=32, depth=32, width=2, caption="Flow\\\\Process1"),
    to_connection("flow_input", "flow_proc1"),
    
    to_Conv("flow_proc2", 28, 16, offset="(2,0,0)", to="(flow_proc1-east)", 
            height=28, depth=28, width=2.5, caption="Flow\\\\Process2"),
    to_connection("flow_proc1", "flow_proc2"),
    
    # ============ 相关性分析模块 ============
    # 静态和动态的相关性分析
    to_Conv("correlation", 32, 64, offset="(3,2,0)", to="(dynamic_proc2-east)", 
            height=32, depth=32, width=4, caption="Correlation\\\\Analysis"),
    to_connection("static_proc2", "correlation"),
    to_connection("dynamic_proc2", "correlation"),
    
    # ============ 掩码生成模块 ============
    # 动态掩码生成分支
    to_Conv("dynamic_mask", 24, 1, offset="(2,1,0)", to="(correlation-east)", 
            height=24, depth=24, width=1, caption="Dynamic\\\\Mask"),
    to_connection("correlation", "dynamic_mask"),
    
    # 光流掩码生成分支
    to_Conv("flow_mask", 24, 1, offset="(4,-1,0)", to="(flow_proc2-east)", 
            height=24, depth=24, width=1, caption="Flow\\\\Mask"),
    to_connection("flow_proc2", "flow_mask"),
    
    # 掩码融合
    to_Conv("fused_mask", 24, 1, offset="(2,0,0)", to="(dynamic_mask-east)", 
            height=24, depth=24, width=1, caption="Fused\\\\Mask"),
    to_connection("dynamic_mask", "fused_mask"),
    to_connection("flow_mask", "fused_mask"),
    
    # ============ 残差融合网络 ============
    # 三流特征融合
    to_Conv("fusion_stage1", 32, 80, offset="(3,-1,0)", to="(correlation-east)", 
            height=32, depth=32, width=5, caption="Fusion\\\\Stage1"),
    to_connection("correlation", "fusion_stage1"),
    to_connection("flow_proc2", "fusion_stage1"),
    
    # 应用掩码
    to_Conv("masked_fusion", 30, 64, offset="(2,0,0)", to="(fusion_stage1-east)", 
            height=30, depth=30, width=4, caption="Masked\\\\Fusion"),
    to_connection("fusion_stage1", "masked_fusion"),
    to_connection("fused_mask", "masked_fusion"),
    
    # 最终融合
    to_Conv("fusion_stage2", 28, 32, offset="(2,0,0)", to="(masked_fusion-east)", 
            height=28, depth=28, width=3, caption="Fusion\\\\Stage2"),
    to_connection("masked_fusion", "fusion_stage2"),
    
    # ============ 最终输出 ============
    to_Conv("final_output", 26, 1, offset="(2,0,0)", to="(fusion_stage2-east)", 
            height=26, depth=26, width=1, caption="Final\\\\Output"),
    to_connection("fusion_stage2", "final_output"),
    
    # ============ 跳跃连接 (类似UNet) ============
    # 从三个输入流到最终输出的跳跃连接
    to_skip("static_input", "final_output", pos=1.3),
    to_skip("dynamic_input", "final_output", pos=1.15),
    to_skip("flow_input", "final_output", pos=1.0),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
