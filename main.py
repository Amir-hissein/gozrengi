"""Primary entrypoint for the modular iris color detector."""

from iris_app.app import main


import sys

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  [BILGI] Program kullanici tarafindan guvenli bir sekilde durduruldu (Ctrl+C).")
        sys.exit(0)
