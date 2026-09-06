"""`python -m mcp_gazette` - the server, independently runnable.

Deliberately a thin shim. Everything is in `server.main` so the entry point can
be imported and tested without a subprocess.
"""

from mcp_gazette.server import main

if __name__ == "__main__":
    raise SystemExit(main())
