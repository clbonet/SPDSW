import os
import requests


def download_bci(path):

    path_folder = os.path.join(path, "data_bci")

    try:
        os.makedirs(path_folder)
    except OSError:
        raise Exception(
            f"{path_folder} already exists. "
            "Please remove the folder"
            ) from FileExistsError

    for subject in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for session in ["E", "T"]:

            url = f"http://bnci-horizon-2020.eu/database/data-sets/001-2014/A0{subject}{session}.mat"
            print(f"Downloading data for subject {subject} - session {session}...")
            response = requests.get(url)

            with open(os.path.join(path_folder, f"A0{subject}{session}.mat"), "wb") as filename:
                filename.write(response.content)

    return path_folder
