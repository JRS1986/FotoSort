"""Entry point: `python -m fotosort ...` or the `fotosort` console script."""
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "review":
        from fotosort.review import main as review_main

        return review_main(argv[1:])
    if argv and argv[0] == "decisions":
        from fotosort.decisions import main as decisions_main

        return decisions_main(argv[1:])
    if argv and argv[0] == "enhance":
        from fotosort.enhance import main as enhance_main

        return enhance_main(argv[1:])
    from fotosort.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
