#setup venv
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
    echo "Venv Created!"
fi

echo "Installing dependencies..."
source "venv/bin/activate" || exit 1

pip install --upgrade pip
pip install -e .
pip freeze > requirements.txt
echo "Setup complete! To activate the venv, run:"
echo "    source ./venv/bin/activate"
