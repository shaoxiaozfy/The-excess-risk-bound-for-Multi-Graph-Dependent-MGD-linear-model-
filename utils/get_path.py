import os

def get_project_path():
    script_path = os.path.abspath(__file__)
    project_dir = os.path.dirname(os.path.dirname(script_path))
    return project_dir + "/"

