"""
FFmpeg路径配置工具
自动检测并配置本地ffmpeg可执行文件路径
"""

import os
import sys
from pathlib import Path
from sorawm.configs import FFMPEG_DIR_PATH
from loguru import logger


def setup_ffmpeg_path():
    """
    配置ffmpeg路径，优先使用本地打包的ffmpeg
    
    查找顺序：
    1. 项目根目录的 ffmpeg/ 目录
    2. 系统环境变量PATH中的ffmpeg
    
    Returns:
        bool: 是否成功配置ffmpeg路径
    """

    # 本地ffmpeg目录
    local_ffmpeg_dir = FFMPEG_DIR_PATH
    
    # 根据操作系统确定可执行文件名
    if sys.platform == "win32":
        ffmpeg_exe = "ffmpeg.exe"
        ffprobe_exe = "ffprobe.exe"
    else:
        ffmpeg_exe = "ffmpeg"
        ffprobe_exe = "ffprobe"
    
    # 检查本地ffmpeg是否存在
    local_ffmpeg_path = local_ffmpeg_dir / ffmpeg_exe
    local_ffprobe_path = local_ffmpeg_dir / ffprobe_exe
    
    if local_ffmpeg_path.exists() and local_ffprobe_path.exists():
        logger.info(f"检测到本地ffmpeg: {local_ffmpeg_dir}")
        
        # 将本地ffmpeg目录添加到PATH环境变量的最前面
        # 这样ffmpeg-python会优先使用本地的ffmpeg
        current_path = os.environ.get("PATH", "")
        ffmpeg_dir_str = str(local_ffmpeg_dir.absolute())
        
        if ffmpeg_dir_str not in current_path:
            os.environ["PATH"] = f"{ffmpeg_dir_str}{os.pathsep}{current_path}"
            logger.info(f"已将ffmpeg目录添加到PATH: {ffmpeg_dir_str}")
        
        # 显式设置FFMPEG和FFPROBE的路径（某些库会使用这些环境变量）
        os.environ["FFMPEG_BINARY"] = str(local_ffmpeg_path.absolute())
        os.environ["FFPROBE_BINARY"] = str(local_ffprobe_path.absolute())
        
        logger.success("FFmpeg配置成功 (使用本地版本)")
        return True
    else:
        logger.warning(
            f"未在本地找到ffmpeg ({local_ffmpeg_dir})，将使用系统安装的ffmpeg"
        )
        logger.info(
            "如需使用便携版，请从以下地址下载ffmpeg并放置到ffmpeg目录："
        )
        logger.info("https://github.com/BtbN/FFmpeg-Builds/releases")
        
        # 检查系统PATH中是否有ffmpeg
        try:
            import shutil
            system_ffmpeg = shutil.which("ffmpeg")
            system_ffprobe = shutil.which("ffprobe")
            
            if system_ffmpeg and system_ffprobe:
                logger.info(f"检测到系统ffmpeg: {system_ffmpeg}")
                return True
            else:
                logger.error("系统中未找到ffmpeg，请安装ffmpeg或将其放置到ffmpeg目录")
                return False
        except Exception as e:
            logger.error(f"检查系统ffmpeg时出错: {e}")
            return False


def check_ffmpeg_available():
    """
    检查ffmpeg是否可用
    
    Returns:
        bool: ffmpeg是否可用
    """
    try:
        import subprocess
        
        # 尝试运行ffmpeg -version
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"FFmpeg不可用: {e}")
        return False


# 在模块导入时自动配置ffmpeg路径
def init_ffmpeg():
    """初始化ffmpeg配置"""
    success = setup_ffmpeg_path()
    if success:
        # 验证ffmpeg是否真的可用
        if check_ffmpeg_available():
            logger.success("✓ FFmpeg已就绪")
        else:
            logger.warning("FFmpeg配置完成但无法正常运行，请检查")
    else:
        logger.warning("FFmpeg未正确配置，视频处理功能可能无法使用")


if __name__ == "__main__":
    # 测试ffmpeg配置
    print("=" * 60)
    print("测试FFmpeg配置")
    print("=" * 60)
    init_ffmpeg()
    print("=" * 60)

