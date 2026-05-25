"""Enable ``python -m homoeogwas`` as a fallback to the ``homoeogwas`` script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
