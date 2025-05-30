# heterogeneous_spacecraft_collaboration/utils/font_utils.py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import platform
import os

def setup_matplotlib_fonts():
    """配置matplotlib以正确显示中文字符"""
    
    # 根据操作系统设置字体路径
    if platform.system() == 'Windows':
        font_dirs = [
            'C:/Windows/Fonts',
            str(Path.home() / 'AppData/Local/Microsoft/Windows/Fonts')
        ]
        font_files = ['SimHei.ttf', 'msyh.ttc', 'simkai.ttf']
    else:
        font_dirs = ['/usr/share/fonts', '/usr/local/share/fonts']
        font_files = ['wqy-microhei.ttc', 'noto-sans-cjk-sc.ttc']

    # 尝试加载中文字体
    font_found = False
    for font_dir in font_dirs:
        for font_file in font_files:
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                try:
                    # 添加字体文件路径
                    fm.fontManager.addfont(font_path)
                    # 设置默认字体
                    plt.rcParams['font.family'] = ['sans-serif']
                    if 'SimHei' in font_file:
                        plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
                    elif 'msyh' in font_file:
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + plt.rcParams['font.sans-serif']
                    elif 'simkai' in font_file:
                        plt.rcParams['font.sans-serif'] = ['KaiTi'] + plt.rcParams['font.sans-serif']
                    elif 'wqy-microhei' in font_file:
                        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
                    elif 'noto-sans-cjk-sc' in font_file:
                        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']
                    
                    print(f"成功加载字体: {font_path}")
                    font_found = True
                    break
                except Exception as e:
                    print(f"加载字体失败 {font_path}: {str(e)}")
                    continue
        if font_found:
            break

    if not font_found:
        print("警告: 未找到合适的中文字体，文字显示可能不正确")

    # 其他必要的matplotlib配置
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['font.size'] = 12
    
    return font_found

if __name__ == '__main__':
    setup_matplotlib_fonts()
    # Test plot
    plt.figure()
    plt.title("中文字体测试图")
    plt.xlabel("X轴标签")
    plt.ylabel("Y轴标签")
    plt.text(0.5, 0.5, "你好，世界！Hello World!", ha='center')
    plt.show()