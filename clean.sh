# Remove all pycache folders and npy files
find . -type f \( -name "*.npy" -o -name "*.tmp" \) -delete
find . -type d -name "__pycache__" -exec rm -r {} +
