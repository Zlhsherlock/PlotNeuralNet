import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

# 动静态光流融合架构 (Dynamic-Static Flow Fusion Architecture)
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # 输入层 - 静态代价空间 Static Cost Volumes
    to_Conv("static_input", 32, 1, offset="(0,0,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Static Cost\\\\[B,1,D,H/4,W/4]"),
    
    # 输入层 - 动态代价空间 Dynamic Cost Volumes  
    to_Conv("dynamic_input", 32, 1, offset="(0,-4,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Dynamic Cost\\\\[B,1,D,H/4,W/4]"),
    
    # 可选光流输入 Flow Input [B,2,H,W]
    to_Conv("flow_input", 64, 2, offset="(0,4,0)", to="(0,0,0)", 
            height=64, depth=64, width=1, caption="Flow Input\\\\[B,2,H,W]"),
    
    # 相关性分析 - 拼接层
    to_Conv("concat_layer", 32, 2, offset="(2,0,0)", to="(static_input-east)", 
            height=32, depth=32, width=2, caption="Concatenation\\\\[B,2,D,H/4,W/4]"),
    
    # 连接输入到拼接层
    to_connection("static_input", "concat_layer"),
    to_connection("dynamic_input", "concat_layer"),
    
    # 3D卷积1 - 相关性分析
    to_Conv("conv3d_1", 32, 16, offset="(1.5,0,0)", to="(concat_layer-east)", 
            height=32, depth=32, width=3, caption="3D Conv\\\\2→16 channels"),
    to_connection("concat_layer", "conv3d_1"),
    
    # 3D卷积2 - 相关性分析
    to_Conv("conv3d_2", 32, 8, offset="(1.5,0,0)", to="(conv3d_1-east)", 
            height=32, depth=32, width=2.5, caption="3D Conv\\\\16→8 channels\\\\corr\\_feat"),
    to_connection("conv3d_1", "conv3d_2"),
    
    # 动态掩码生成分支
    to_Conv("dynamic_mask_gen", 28, 1, offset="(2,2,0)", to="(conv3d_2-east)", 
            height=28, depth=28, width=1, caption="Dynamic Mask\\\\Generation"),
    to_connection("conv3d_2", "dynamic_mask_gen"),
    
    # 光流掩码生成分支
    to_Conv("flow_magnitude", 32, 1, offset="(4,-2,0)", to="(flow_input-east)", 
            height=32, depth=32, width=1, caption="Flow Magnitude\\\\sqrt(F\\_x²+F\\_y²)"),
    to_connection("flow_input", "flow_magnitude"),
    
    to_Conv("flow_soft_mask", 28, 1, offset="(1.5,0,0)", to="(flow_magnitude-east)", 
            height=28, depth=28, width=1, caption="Flow Soft Mask\\\\sigmoid((mag-1.0)×5.0)"),
    to_connection("flow_magnitude", "flow_soft_mask"),
    
    to_Conv("flow_3d_ext", 28, 1, offset="(1.5,0,0)", to="(flow_soft_mask-east)", 
            height=28, depth=28, width=1, caption="3D Extension\\\\flow\\_mask\\_3d"),
    to_connection("flow_soft_mask", "flow_3d_ext"),
    
    # 掩码组合
    to_Conv("mask_combine", 28, 1, offset="(2,0,0)", to="(dynamic_mask_gen-east)", 
            height=28, depth=28, width=1, caption="Mask Combination\\\\max(dynamic\\_mask,\\\\flow\\_mask)"),
    to_connection("dynamic_mask_gen", "mask_combine"),
    to_connection("flow_3d_ext", "mask_combine"),
    
    # 残差融合网络 - 多信息拼接
    to_Conv("multi_concat", 32, 11, offset="(2,-1,0)", to="(conv3d_2-east)", 
            height=32, depth=32, width=4, caption="Multi-info Concat\\\\cat([C\\_s, C\\_d, corr\\_feat,\\\\mask]) [B,11,D,H/4,W/4]"),
    to_connection("conv3d_2", "multi_concat"),
    to_connection("mask_combine", "multi_concat"),
    
    # 残差融合网络 - 3D卷积序列
    to_Conv("fusion_conv1", 32, 32, offset="(2,0,0)", to="(multi_concat-east)", 
            height=32, depth=32, width=5, caption="Residual Fusion\\\\3D Conv 11→32"),
    to_connection("multi_concat", "fusion_conv1"),
    
    to_Conv("fusion_conv2", 30, 16, offset="(1.5,0,0)", to="(fusion_conv1-east)", 
            height=30, depth=30, width=4, caption="3D Conv\\\\32→16"),
    to_connection("fusion_conv1", "fusion_conv2"),
    
    to_Conv("fusion_conv3", 28, 1, offset="(1.5,0,0)", to="(fusion_conv2-east)", 
            height=28, depth=28, width=1, caption="3D Conv\\\\16→1\\\\C\\_d"),
    to_connection("fusion_conv2", "fusion_conv3"),
    
    # 最终输出 - 融合结果
    to_Conv("final_output", 28, 11, offset="(2,0,0)", to="(fusion_conv3-east)", 
            height=28, depth=28, width=4, caption="Final Output\\\\[B,11,D,H/4,W/4]\\\\Multi-info Features"),
    to_connection("fusion_conv3", "final_output"),
    to_connection("static_input", "final_output"),
    to_connection("mask_combine", "final_output"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
