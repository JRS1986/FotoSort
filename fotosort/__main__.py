import sys

if len(sys.argv) > 1 and sys.argv[1] == "enhance":
    from fotosort.enhance import main

    sys.exit(main(sys.argv[2:]))
else:
    from fotosort.cli import main

    sys.exit(main())
