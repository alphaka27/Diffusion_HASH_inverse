from IPython import get_ipython

def is_notebook():
    shell = get_ipython()

    if shell is None:
        return False

    shell_name = shell.__class__.__name__

    return shell_name == "ZMQInteractiveShell"

if is_notebook():
    print("Notebook/Jupyter 환경")
else:
    print("CLI 환경")