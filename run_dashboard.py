import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard.app import main
main()
