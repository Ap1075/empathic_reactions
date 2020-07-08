echo "" &&
echo "Running experiments:" &&
cd modeling &&
echo "....Running cross-validation." &&
# cd main/crossvalidation && python experiment.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python experiment_orig.py && python ttest.py > ttest.txt &&
cd main/crossvalidation && python train_classification.py && python ttest.py > ttest.txt &&
cd ../.. &&
cd .. && echo "Experiments completed." && echo ""
