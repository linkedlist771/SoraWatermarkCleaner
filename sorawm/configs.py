from pathlib import Path

ROOT = Path(__file__).parent.parent


RESOURCES_DIR = ROOT / "resources"
WATER_MARK_TEMPLATE_IMAGE_PATH = RESOURCES_DIR / "watermark_template.png"

WATER_MARK_DETECT_YOLO_WEIGHTS = RESOURCES_DIR / "best.pt"

WATER_MARK_DETECT_YOLO_WEIGHTS_HASH_JSON = RESOURCES_DIR / "model_version.json"


CHECKPOINT_DIR = RESOURCES_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
SPYNET_CHECKPOINT_PATH = CHECKPOINT_DIR / "spynet_20210409-c6c1bd09.pth"
# release_model/E2FGVI-HQ-CVPR22.pth
E2FGVI_HQ_CHECKPOINT_PATH = CHECKPOINT_DIR / "E2FGVI-HQ-CVPR22.pth"
E2FGVI_HQ_CHECKPOINT_REMOTE_URL = "https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/E2FGVI-HQ-CVPR22.pth"

PHY_NET_CHECKPOINT_REMOTE_URL = "https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth"

PHY_NET_CHECKPOINT_PATH = CHECKPOINT_DIR / "spynet_20210409-c6c1bd09.pth"

OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


DEFAULT_WATERMARK_REMOVE_MODEL = "lama"

WORKING_DIR = ROOT / "working_dir"
WORKING_DIR.mkdir(exist_ok=True, parents=True)

LOGS_PATH = ROOT / "logs"
LOGS_PATH.mkdir(exist_ok=True, parents=True)

DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True, parents=True)

SQLITE_PATH = DATA_PATH / "db.sqlite3"

FRONTUI_DIR = ROOT / "frontend"
FRONTUI_DIR.mkdir(exist_ok=True, parents=True)

FRONTUI_DIST_DIR = FRONTUI_DIR / "dist"
FRONTUI_DIST_DIR.mkdir(exist_ok=True, parents=True)

FRONTUI_DIST_DIR_ASSETS = FRONTUI_DIST_DIR / "assets"
FRONTUI_DIST_DIR_ASSETS.mkdir(exist_ok=True, parents=True)

FRONTUI_DIST_DIR_INDEX_HTML = FRONTUI_DIST_DIR / "index.html"


## TORCH COMPILE

ENABLE_E2FGVI_HQ_TORCH_COMPILE = True

TORCH_COMPILE_DIR = Path("~").expanduser() / ".cache" / "torch_compile"
TORCH_COMPILE_DIR.mkdir(exist_ok=True, parents=True)

E2FGVI_HQ_TORCH_COMPILE_DIR = TORCH_COMPILE_DIR / "e2fgvi_hq"
E2FGVI_HQ_TORCH_COMPILE_DIR.mkdir(exist_ok=True, parents=True)

E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS = E2FGVI_HQ_TORCH_COMPILE_DIR / "artifacts.bin"
E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS_BF16 = (
    E2FGVI_HQ_TORCH_COMPILE_DIR / "artifacts_bf16.bin"
)
# artifacts


## YOLO BATCH SIZE
DEFAULT_DETECT_BATCH_SIZE = 8
