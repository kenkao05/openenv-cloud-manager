import uvicorn
from env import app  # noqa: F401 — re-export for OpenEnv discovery


def main():
    uvicorn.run("env:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
