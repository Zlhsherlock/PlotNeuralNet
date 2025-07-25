import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

# 动静态光流融合架构 (Dynamic-Static Flow Fusion Architecture)
# 使用官方支持的层类型重新设计
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # 1. 输入层组 - 三个输入
    # 静态代价空间 Static Cost Volumes [B,1,D,H/4,W/4]
    to_Conv("static_cv", 32, 1, offset="(0,0,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Static CV"),
    
    # 动态代价空间 Dynamic Cost Volumes [B,1,D,H/4,W/4]
    to_Conv("dynamic_cv", 32, 1, offset="(0,-3,0)", to="(0,0,0)", 
            height=32, depth=32, width=1, caption="Dynamic CV"),
    
    # 光流输入 Flow Input [B,2,H,W]
    to_Conv("flow_input", 64, 2, offset="(0,3,0)", to="(0,0,0)", 
            height=64, depth=64, width=1, caption="Flow Input"),
    
    # 2. 相关性分析模块 - Correlation Analysis
    # 拼接静态和动态代价空间 -> [B,2,D,H/4,W/4]
    to_Conv("corr_concat", 32, 2, offset="(2.5,0,0)", to="(static_cv-east)", 
            height=32, depth=32, width=2, caption="Concat"),
    to_connection("static_cv", "corr_concat"),
    to_connection("dynamic_cv", "corr_concat"),
    
    # 3D卷积提取相关性特征
    to_Conv("corr_conv1", 32, 16, offset="(1.5,0,0)", to="(corr_concat-east)", 
            height=32, depth=32, width=3, caption="3D Conv 2→16"),
    to_connection("corr_concat", "corr_conv1"),
    
    to_Conv("corr_feat", 32, 8, offset="(1.5,0,0)", to="(corr_conv1-east)", 
            height=32, depth=32, width=2.5, caption="Corr Features"),
    to_connection("corr_conv1", "corr_feat"),
    
    # 3. 掩码生成模块 - Mask Generation
    # 动态掩码分支
    to_Conv("dyn_mask", 28, 1, offset="(2,1.5,0)", to="(corr_feat-east)", 
            height=28, depth=28, width=1, caption="Dynamic Mask"),
    to_connection("corr_feat", "dyn_mask"),
    
    # 光流掩码分支
    to_Conv("flow_mag", 32, 1, offset="(4,-2,0)", to="(flow_input-east)", 
            height=32, depth=32, width=1, caption="Flow Magnitude"),
    to_connection("flow_input", "flow_mag"),
    
    to_Conv("flow_mask", 28, 1, offset="(1.5,0,0)", to="(flow_mag-east)", 
            height=28, depth=28, width=1, caption="Flow Mask"),
    to_connection("flow_mag", "flow_mask"),
    
    # 掩码组合
    to_Conv("final_mask", 28, 1, offset="(2,-0.5,0)", to="(dyn_mask-east)", 
            height=28, depth=28, width=1, caption="Final Mask"),
    to_connection("dyn_mask", "final_mask"),
    to_connection("flow_mask", "final_mask"),
    
    # 4. 残差融合网络 - Residual Fusion Network
    # 多信息拼接: corr_feat(8) + static_cv(1) + dynamic_cv(1) + mask(1) = 11通道
    to_Conv("fusion_input", 32, 11, offset="(2.5,-1,0)", to="(corr_feat-east)", 
            height=32, depth=32, width=4, caption="Fusion Input"),
    to_connection("corr_feat", "fusion_input"),
    to_connection("final_mask", "fusion_input"),
    
    # 残差融合网络的卷积序列
    to_Conv("fusion_conv1", 32, 32, offset="(2,0,0)", to="(fusion_input-east)", 
            height=32, depth=32, width=5, caption="Fusion Conv"),
    to_connection("fusion_input", "fusion_conv1"),
    
    to_Conv("fusion_conv2", 30, 16, offset="(1.5,0,0)", to="(fusion_conv1-east)", 
            height=30, depth=30, width=4, caption="32→16"),
    to_connection("fusion_conv1", "fusion_conv2"),
    
    to_Conv("fused_cv", 28, 1, offset="(1.5,0,0)", to="(fusion_conv2-east)", 
            height=28, depth=28, width=1, caption="Fused CV"),
    to_connection("fusion_conv2", "fused_cv"),
    
    # 5. 最终输出 - Final Multi-info Output
    # 拼接所有关键特征: static_cv(1) + fused_cv(1) + corr_feat(8) + mask(1) = 11通道
    to_Conv("final_output", 28, 11, offset="(2,0,0)", to="(fused_cv-east)", 
            height=28, depth=28, width=4, caption="Final Output"),
    to_connection("fused_cv", "final_output"),
    
    # 跳跃连接到最终输出
    to_skip("static_cv", "final_output", pos=1.2),
    to_skip("final_mask", "final_output", pos=1.1),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
