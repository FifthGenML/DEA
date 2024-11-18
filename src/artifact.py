import os
import requests
import src.config as config


def create_artifact():
    SAVE_DIR = "downloaded_artifacts"
    os.makedirs(SAVE_DIR, exist_ok=True)  

    for artifact in config.ARTIFACT_FILES:
        url = f"{config.CRUCIBLE_URL}/api/artifacts/{config.CHALLENGE}/{artifact}"
        headers = {"X-API-Key": config.CRUCIBLE_API_KEY}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            artifact_path = os.path.join(SAVE_DIR, artifact)
            with open(artifact_path, "wb") as file:
                file.write(response.content)
            print(f"{artifact} was successfully downloaded to {artifact_path}")
        else:
            print(f"Failed to download {artifact} - Status code: {response.status_code}")
    return artifact_path