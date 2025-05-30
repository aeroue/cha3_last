# experiments/fix_chinese.py
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def setup_chinese_fonts():
    """设置中文字体支持"""
    import matplotlib.font_manager as fm
    
    # 尝试找到系统中存在的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'DengXian', 
                     'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']
    
    # 检查是否有自定义字体目录
    custom_font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts')
    if os.path.exists(custom_font_dir):
        # 添加自定义字体目录
        font_files = [f for f in os.listdir(custom_font_dir) if f.endswith(('.ttf', '.otf'))]
        for font_file in font_files:
            fm.fontManager.addfont(os.path.join(custom_font_dir, font_file))
        print(f"已添加自定义字体目录: {custom_font_dir}")
    
    # 查找系统字体
    chinese_font_found = False
    for font in chinese_fonts:
        try:
            font_path = fm.findfont(font, fallback_to_default=False)
            if font_path:
                matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
                chinese_font_found = True
                print(f"使用中文字体: {font}")
                break
        except:
            continue
    
    if not chinese_font_found:
        print("警告: 未找到合适的中文字体，尝试使用备用方法...")
        
        # 备用方法1: 设置多个候选字体，希望matplotlib能找到一个有效的
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['Arial', 'DejaVu Sans']
        
        # 备用方法2: 创建一个简单的文本测试
        try:
            fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print("中文字体测试通过")
        except Exception as e:
            print(f"中文字体测试失败: {e}")
    
    # 确保负号能正常显示
    matplotlib.rcParams['axes.unicode_minus'] = False