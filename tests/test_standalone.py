"""The standalone package must not require kidsdata at import time."""

import os
import subprocess
import sys
from pathlib import Path


def test_importing_ftsdata_does_not_import_kidsdata():
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project), env.get("PYTHONPATH")]))

    code = (
        "import sys; import ftsdata; "
        "assert 'kidsdata' not in sys.modules, "
        "sorted(name for name in sys.modules if name.startswith('kidsdata'))"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
