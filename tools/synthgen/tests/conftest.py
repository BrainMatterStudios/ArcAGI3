import sys
from pathlib import Path

SYNTHGEN = Path(__file__).resolve().parents[1]
if str(SYNTHGEN) not in sys.path:
    sys.path.insert(0, str(SYNTHGEN))
