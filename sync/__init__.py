from .sync import main as _main


def run() -> int:
    return _main(["sync"])