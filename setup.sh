#setup venv
if [ ! -d "venv" ]; then
    echo -e "Creating venv"
    python3 -m venv venv
    echo -e "Venv Created"
fi
#actviate

source "$(pwd)/venv/bin/activate"

echo -e "activating"
pip install --upgrade pip
pip install -e .
pip freeze > requirements.txt
echo -e "activated"
