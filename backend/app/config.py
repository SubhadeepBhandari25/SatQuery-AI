import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = BASE_DIR / 'runtime'

# Runtime directories for strictly file-based temporary storage (NO DATABASE)
UPLOADS_DIR = RUNTIME_DIR / 'uploads'
RESULTS_DIR = RUNTIME_DIR / 'results'
OVERLAYS_DIR = RUNTIME_DIR / 'overlays'
REPORTS_DIR = RUNTIME_DIR / 'reports'
TEMP_DIR = RUNTIME_DIR / 'temp'
CACHE_DIR = RUNTIME_DIR / 'cache'

for directory in [UPLOADS_DIR, RESULTS_DIR, OVERLAYS_DIR, REPORTS_DIR, TEMP_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware Awareness
DEVICE_ENV = os.getenv('DEVICE', 'auto').lower()
if DEVICE_ENV == 'cuda' and torch.cuda.is_available():
    DEVICE = 'cuda'
elif DEVICE_ENV == 'cpu':
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Upload and Security Configuration
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '150'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
ALLOWED_MIME_TYPES = {
    'image/tiff',
    'image/x-tiff',
    'image/png',
    'image/jpeg',
    'image/bmp',
    'application/octet-stream',
}

SESSION_RETENTION_HOURS = int(os.getenv('SESSION_RETENTION_HOURS', '24'))
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', '8008'))
