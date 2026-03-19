CHUNK_SIZE_PER_GB_VRAM = 5  # 1GB can process about 5 frames in chunk size
# Cap adapted_chunk_size to keep inner_chunk_size <= 10 (2 model_forward calls per inner chunk)
# inner_chunk_size = 0.2 * adapted_chunk_size; for 2 calls: inner <= 10 -> adapted <= 50
MAX_ADAPTED_CHUNK_SIZE = 50
