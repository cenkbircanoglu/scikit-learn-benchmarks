#!/usr/bin/env bash

cd ./$1

for output_size in 8 16 32 64 128
do
    for dim_red in factoranalysis fastica pca nmf sparsepca truncatedsvd
    do
        for notebook in ${dim_red}_${output_size}.ipynb
        do
            /Users/cenk/.virtualenvs/experiments_on_mnist/bin/jupyter nbconvert --execute --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1 \
                --to notebook --inplace $notebook
        done
    done
done

/Users/cenk/.virtualenvs/experiments_on_mnist/bin/jupyter nbconvert --execute --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1 \
            --to notebook --inplace base.ipynb