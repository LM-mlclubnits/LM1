import sys
import streamlit.web.cli as stcli


def main():
    sys.argv = ["streamlit", "run", "app/streamlitApp.py"]
    sys.exit(stcli.main())


if __name__ == '__main__':
    main()
