def to_project_root(*args):
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', *args))
