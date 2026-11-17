"""``python -m aifirm`` runs the command-line interface (same as the ``aifirm`` script)."""

import sys

from .cli import main

sys.exit(main())
