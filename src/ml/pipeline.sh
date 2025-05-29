source venv/bin/activate
python3 $1
echo "Finished generating synthetic data"
jupyter nbconvert --to notebook --execute $2 --output "executed_{$2}"
echo "Finished training, testing and validating model"