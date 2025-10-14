"""
测试FFmpeg配置脚本
用于验证本地FFmpeg是否正确配置
"""

from sorawm.utils.ffmpeg_utils import init_ffmpeg, check_ffmpeg_available

if __name__ == "__main__":
    print("=" * 70)
    print("FFmpeg配置测试")
    print("=" * 70)
    
    # 初始化ffmpeg
    init_ffmpeg()
    
    print("\n" + "=" * 70)
    print("检查FFmpeg可用性")
    print("=" * 70)
    
    # 检查ffmpeg是否可用
    if check_ffmpeg_available():
        print("\n✓ 测试通过！FFmpeg已正确配置并可以使用")
    else:
        print("\n✗ 测试失败！FFmpeg无法正常工作")
        print("请检查：")
        print("1. ffmpeg/目录下是否有ffmpeg.exe和ffprobe.exe")
        print("2. 或者系统PATH中是否安装了ffmpeg")
    
    print("=" * 70)

