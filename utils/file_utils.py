import os


def check_path_exists(path: str):
    """Checks whether the specified path exists and creates it if not"""
    # Check if directory exists
    directory = os.path.dirname(path)
    if directory == "":
        directory = "."
    dir_exist = os.path.exists(directory)
    if not dir_exist:
        # Create a new directory because it does not exist
        os.makedirs(directory)
        print(f"The new directory, {directory}, was created.")