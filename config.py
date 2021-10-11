import os


class Project:
    project_path = os.path.dirname(os.path.abspath(__file__))

    gait_dataset = "%s\dataset" % project_path


