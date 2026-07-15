import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import finetune_worker
import validation_worker
import densenet_worker


def main():
    if len(sys.argv) < 2:
        print("Usage: unified_worker <mode> <args...>")
        print("Modes: finetune, validation, densenet")
        sys.exit(1)

    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "finetune":
        finetune_worker.main()
    elif mode == "validation":
        validation_worker.main()
    elif mode == "densenet":
        densenet_worker.main()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
