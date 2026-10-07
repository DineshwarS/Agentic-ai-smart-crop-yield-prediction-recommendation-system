# backend package
import sys
from pathlib import Path

# Ensure backend directory is always in sys.path for direct imports like 'app.main'
_BACKEND_DIR = Path(__file__).parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
