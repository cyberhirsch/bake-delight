"""Entry point of the packaged scan job (BakeDelightJob.exe), which the interface starts for each
stage: scanopt's command line. The spec puts scanopt/ first on the path, so `scanopt` is scanopt.py."""
import sys

import scanopt

sys.exit(scanopt.main())
