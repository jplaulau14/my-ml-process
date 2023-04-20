import os

def get_data_path() -> str:
    """
    Get the path to the data directory.

    Returns
    -------
    str
        Path to the data directory.
    """
    # Get the path to the current file
    current_file = os.path.abspath(__file__)

    # Go up the directory tree until we find the project root
    project_root = os.path.dirname(current_file)
    while not os.path.exists(os.path.join(project_root, 'data')):
        project_root = os.path.dirname(project_root)

    # Construct the path to the data directory
    data_path = os.path.join(project_root, 'data')

    return data_path

def get_model_path() -> str:
    """
    Get the path to the model directory.

    Returns
    -------
    str
        Path to the model directory.
    """
    # Get the path to the current file
    current_file = os.path.abspath(__file__)

    # Go up the directory tree until we find the project root
    project_root = os.path.dirname(current_file)
    while not os.path.exists(os.path.join(project_root, 'models')):
        project_root = os.path.dirname(project_root)

    # Construct the path to the model directory
    model_path = os.path.join(project_root, 'models')

    return model_path