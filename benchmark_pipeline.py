"""
Performance benchmark script for comparing serial (run) vs pipeline (run_overlap) execution.
"""

import time
from pathlib import Path
from loguru import logger
from sorawm.core import SoraWM


def benchmark_method(method_name: str, sora_wm: SoraWM, input_path: Path, output_path: Path):
    """
    Benchmark a single method.
    
    Args:
        method_name: Method name ('run' or 'run_overlap')
        sora_wm: SoraWM instance
        input_path: Input video path
        output_path: Output video path
    
    Returns:
        Execution time in seconds
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting benchmark: {method_name}")
    logger.info(f"{'='*60}")
    
    # Ensure output file doesn't exist
    if output_path.exists():
        output_path.unlink()
    
    # Get method
    method = getattr(sora_wm, method_name)
    
    # Start timing
    start_time = time.time()
    
    try:
        method(input_path, output_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"\n{method_name} completed!")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Output file: {output_path}")
        
        return elapsed_time
        
    except Exception as e:
        logger.error(f"{method_name} execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function - run performance benchmark"""
    
    # ========= Configuration =========
    # Input video path (use shorter video for quick testing)
    input_video_path = Path("resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4")
    
    # Check if input file exists
    if not input_video_path.exists():
        logger.error(f"Input video not found: {input_video_path}")
        logger.info("Available videos:")
        resources_dir = Path("resources")
        if resources_dir.exists():
            for video in resources_dir.glob("*.mp4"):
                logger.info(f"  - {video}")
        return
    
    # Output paths
    output_dir = Path("outputs/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_serial = output_dir / "benchmark_serial.mp4"
    output_parallel = output_dir / "benchmark_parallel.mp4"
    
    # =======================
    
    logger.info("\n" + "="*60)
    logger.info("Performance Benchmark Test Starting")
    logger.info("="*60)
    logger.info(f"Input video: {input_video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Create SoraWM instance
    logger.info("Initializing SoraWM...")
    sora_wm = SoraWM()
    
    # Test 1: Serial execution (original method)
    time_serial = benchmark_method("run", sora_wm, input_video_path, output_serial)
    
    # Wait a bit to ensure resources are released
    time.sleep(2)
    
    # Test 2: Pipeline parallel execution
    time_parallel = benchmark_method("run_overlap", sora_wm, input_video_path, output_parallel)
    
    # Performance comparison summary
    logger.info("\n" + "="*60)
    logger.info("Performance Comparison Summary")
    logger.info("="*60)
    
    if time_serial and time_parallel:
        logger.info(f"Serial execution (run):           {time_serial:.2f} seconds")
        logger.info(f"Pipeline parallel (run_overlap):  {time_parallel:.2f} seconds")
        logger.info("")
        
        # Calculate performance improvement
        speedup = time_serial / time_parallel
        improvement = ((time_serial - time_parallel) / time_serial) * 100
        
        if speedup > 1:
            logger.info(f"✅ Performance improvement: {speedup:.2f}x speedup")
            logger.info(f"✅ Time reduction: {improvement:.1f}%")
            logger.info(f"✅ Time saved: {time_serial - time_parallel:.2f} seconds")
        elif speedup < 1:
            logger.warning(f"⚠️  Performance degradation: {1/speedup:.2f}x slower")
            logger.warning(f"⚠️  Time increase: {-improvement:.1f}%")
        else:
            logger.info("Performance equivalent")
    else:
        logger.error("Benchmark failed, cannot compare performance")
    
    logger.info("\n" + "="*60)
    logger.info("Benchmark Complete")
    logger.info("="*60)


if __name__ == "__main__":
    main()

