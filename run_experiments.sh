echo "" &&
echo "Running experiments:" &&
cd modeling &&
echo "....Running cross-validation." &&
# cd main/crossvalidation && python experiment.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python experiment_orig.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python train_regression.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python emp_regression.py && python ttest.py > ttest.txt &&
cd main/crossvalidation && python simultaneous_models.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python hindi_exp.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python test_para.py && python ttest.py > ttest.txt &&
# cd main/crossvalidation && python train_classification.py && python ttest.py > ttest.txt &&
cd ../.. &&
cd .. && echo "Experiments completed." && echo ""
