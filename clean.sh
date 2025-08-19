# Remove all pycache folders and data files
find . -type f \( -name "*.npy" -o -name "*.pt" -o -name "*.tmp" \) -delete
find . -type d -name "__pycache__" -exec rm -r {} +
