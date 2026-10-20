from .cleaner import main as _main


def run() -> int:
    return _main(["cleaner"])   # no path given, so it uses the latest raw file