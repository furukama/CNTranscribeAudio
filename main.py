from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "8011"))
    uvicorn.run("app.server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
