import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

# 动静态光流三流融合架构 (Dynamic-Static-Flow Three-Stream Fusion Architecture)
# 参考UNet风格的清晰三流设计
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # ==================== 三个输入流 ====================
    
    # 第一流：静态代价空间流 (上方流)
    to_Conv("static_input", 32, 1, offset="(0,5,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Static Cost\\\\Volume"),
    
    to_Conv("static_proc1", 32, 16, offset="(2,0,0)", to="(static_input-east)", 
            height=32, depth=32, width=2, caption="Static\\\\Process1"),
    to_connection("static_input", "static_proc1"),
    
    to_Conv("static_proc2", 28, 32, offset="(2,0,0)", to="(static_proc1-east)", 
            height=28, depth=28, width=3, caption="Static\\\\Process2"),
    to_connection("static_proc1", "static_proc2"),
    
    # 第二流：动态代价空间流 (中间流)  
    to_Conv("dynamic_input", 32, 1, offset="(0,0,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Dynamic Cost\\\\Volume"),
    
    to_Conv("dynamic_proc1", 32, 16, offset="(2,0,0)", to="(dynamic_input-east)", 
            height=32, depth=32, width=2, caption="Dynamic\\\\Process1"),
    to_connection("dynamic_input", "dynamic_proc1"),
    
    to_Conv("dynamic_proc2", 28, 32, offset="(2,0,0)", to="(dynamic_proc1-east)", 
            height=28, depth=28, width=3, caption="Dynamic\\\\Process2"),
    to_connection("dynamic_proc1", "dynamic_proc2"),
    
    # 第三流：光流输入流 (下方流)
    to_Conv("flow_input", 64, 2, offset="(0,-5,0)", to="(0,0,0)", 
            height=64, depth=64, width=1, caption="Optical Flow\\\\Input"),
    
    to_Conv("flow_proc1", 32, 8, offset="(2,0,0)", to="(flow_input-east)", 
            height=32, depth=32, width=2, caption="Flow\\\\Process1"),
    to_connection("flow_input", "flow_proc1"),
    
    to_Conv("flow_proc2", 28, 16, offset="(2,0,0)", to="(flow_proc1-east)", 
            height=28, depth=28, width=2.5, caption="Flow\\\\Process2"),
    to_connection("flow_proc1", "flow_proc2"),
    
    # ==================== 相关性分析阶段 ====================
    
    # 三流特征拼接
    to_Conv("correlation_concat", 32, 80, offset="(3,0,0)", to="(dynamic_proc2-east)", 
            height=32, depth=32, width=5, caption="Correlation\\\\Analysis\\\\Concat[32+32+16]"),
    
    # 连接三个流到相关性分析
    to_connection("static_proc2", "correlation_concat"),
    to_connection("dynamic_proc2", "correlation_concat"),  
    to_connection("flow_proc2", "correlation_concat"),
    
    # 3D卷积处理相关性
    to_Conv("correlation_3d1", 30, 64, offset="(2,0,0)", to="(correlation_concat-east)", 
            height=30, depth=30, width=4, caption="3D Conv1\\\\80→64"),
    to_connection("correlation_concat", "correlation_3d1"),
    
    to_Conv("correlation_3d2", 28, 32, offset="(1.5,0,0)", to="(correlation_3d1-east)", 
            height=28, depth=28, width=3, caption="3D Conv2\\\\64→32"),
    to_connection("correlation_3d1", "correlation_3d2"),
    
    # ==================== 掩码生成分支 ====================
    
    # 动态掩码生成 (从相关性特征)
    to_Conv("dynamic_mask", 24, 1, offset="(1,3,0)", to="(correlation_3d2-east)", 
            height=24, depth=24, width=1, caption="Dynamic\\\\Mask\\\\Generation"),
    to_connection("correlation_3d2", "dynamic_mask"),
    
    # 光流掩码生成 (从光流特征)
    to_Conv("flow_mask_gen", 24, 1, offset="(8,-3,0)", to="(flow_proc2-east)", 
            height=24, depth=24, width=1, caption="Flow Mask\\\\Generation"),
    to_connection("flow_proc2", "flow_mask_gen"),
    
    # 掩码组合
    to_Conv("mask_fusion", 22, 1, offset="(2,0,0)", to="(dynamic_mask-east)", 
            height=22, depth=22, width=1, caption="Mask\\\\Combination\\\\max(Dyn,Flow)"),
    to_connection("dynamic_mask", "mask_fusion"),
    to_connection("flow_mask_gen", "mask_fusion"),
    
    # ==================== 残差融合网络 ====================
    
    # 多信息融合输入 (相关性特征 + 掩码)
    to_Conv("residual_input", 28, 33, offset="(2,0,0)", to="(correlation_3d2-east)", 
            height=28, depth=28, width=4, caption="Residual Input\\\\[32+1 channels]"),
    to_connection("correlation_3d2", "residual_input"),
    to_connection("mask_fusion", "residual_input"),
    
    # 残差融合层1
    to_Conv("residual_conv1", 26, 64, offset="(2,0,0)", to="(residual_input-east)", 
            height=26, depth=26, width=4, caption="Residual\\\\Conv1\\\\33→64"),
    to_connection("residual_input", "residual_conv1"),
    
    # 残差融合层2
    to_Conv("residual_conv2", 24, 32, offset="(1.5,0,0)", to="(residual_conv1-east)", 
            height=24, depth=24, width=3, caption="Residual\\\\Conv2\\\\64→32"),
    to_connection("residual_conv1", "residual_conv2"),
    
    # 残差融合层3
    to_Conv("residual_conv3", 22, 16, offset="(1.5,0,0)", to="(residual_conv2-east)", 
            height=22, depth=22, width=2, caption="Residual\\\\Conv3\\\\32→16"),
    to_connection("residual_conv2", "residual_conv3"),
    
    # ==================== 最终输出 ====================
    
    # 融合输出 (包含丰富特征信息)
    to_Conv("final_features", 24, 49, offset="(2,0,0)", to="(residual_conv3-east)", 
            height=24, depth=24, width=4, caption="Multi-info\\\\Features\\\\[1+1+2+16+32-3]"),
    to_connection("residual_conv3", "final_features"),
    
    # 最终预测输出
    to_Conv("final_output", 22, 1, offset="(2,0,0)", to="(final_features-east)", 
            height=22, depth=22, width=1, caption="Final\\\\Output\\\\Prediction"),
    to_connection("final_features", "final_output"),
    
    # ==================== 跳跃连接 ====================
    
    # 从三个输入流到最终特征的跳跃连接
    to_skip("static_input", "final_features", pos=1.8),
    to_skip("dynamic_input", "final_features", pos=1.5), 
    to_skip("flow_input", "final_features", pos=1.2),
    
    # 从相关性分析到最终特征的跳跃连接
    to_skip("correlation_3d2", "final_features", pos=1.1),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
