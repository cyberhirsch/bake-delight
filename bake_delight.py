"""Entry point for the bake-delight CLI and MCP server; works from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bakedelight.cli import main  # noqa: E402

sys.exit(main())
