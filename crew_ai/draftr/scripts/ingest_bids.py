#!/usr/bin/env python
"""
Thin shim for direct execution without installing the package.

    python scripts/ingest_bids.py [flags]

The canonical implementation lives in src/draftr/scripts/ingest_bids.py
and is also available as the installed CLI command `ingest_bids`.
"""
import sys
from pathlib import Path

# Make the package importable when running this script directly in a dev environment.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from draftr.scripts.ingest_bids import parse_args, main  # noqa: E402

if __name__ == "__main__":
    args = parse_args()
    main(
        bids_dir=Path(args.bids_dir) if args.bids_dir else None,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
        config_path=Path(args.config),
    )
