#!/usr/bin/env bash

for N_COMPONENTS in 8 16 32 64 128
do
    export N_COMPONENTS=$N_COMPONENTS
    for DIMENSIONALITY_ALGORITHM in factor_analysis fact_ica nmf pca sparse_pca truncated_svd
    do
        export DIMENSIONALITY_ALGORITHM=$DIMENSIONALITY_ALGORITHM
        for DATABASE in breast_cancer cov_type digits iris kddcup99 lfw mnist news_groups olivetti_faces rcv1 wine
        do
            export DATABASE=$DATABASE
            mkdir outputs/${DATABASE}
            jupyter nbconvert --execute base_notebook.ipynb \
                --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1  --to notebook --output \
                outputs/${DATABASE}/${N_COMPONENTS}_${DIMENSIONALITY_ALGORITHM}.ipynb
        done
    done
done

export N_COMPONENTS=false
export DIMENSIONALITY_ALGORITHM=false
for DATABASE in breast_cancer cov_type digits iris kddcup99 lfw mnist news_group olivetti_faces rcv1 wine
do
    export DATABASE=$DATABASE
    jupyter nbconvert --execute base_notebook.ipynb \
        --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1  --to notebook --output \
        outputs/${DATABASE}.ipynb
done
