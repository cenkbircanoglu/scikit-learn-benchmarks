#!/usr/bin/env bash


for DATABASE in breast_cancer olivetti_faces digits iris mnist wine lfw rcv1 news_groups cov_type kddcup99
do
    export DATABASE=$DATABASE
    echo $DATABASE
    mkdir outputs/${DATABASE}
    for N_COMPONENTS in 8 16 32 64 128
    do
        export N_COMPONENTS=$N_COMPONENTS
        echo $N_COMPONENTS
        for DIMENSIONALITY_ALGORITHM in factor_analysis fast_ica nmf pca sparse_pca truncated_svd
        do
            export DIMENSIONALITY_ALGORITHM=$DIMENSIONALITY_ALGORITHM
            echo $DIMENSIONALITY_ALGORITHM
            if [ ! -f results/${DATABASE}_${N_COMPONENTS}_${DIMENSIONALITY_ALGORITHM}.csv ]; then
                jupyter nbconvert --execute base_notebook.ipynb \
                    --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1  --to notebook --output \
                    outputs/${DATABASE}/${N_COMPONENTS}_${DIMENSIONALITY_ALGORITHM}.ipynb
            fi
        done
    done
done

export N_COMPONENTS=false
export DIMENSIONALITY_ALGORITHM=false
for DATABASE in breast_cancer olivetti_faces digits iris mnist wine lfw rcv1 news_groups cov_type kddcup99
do
    export DATABASE=$DATABASE
    jupyter nbconvert --execute base_notebook.ipynb \
        --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1  --to notebook --output \
        outputs/${DATABASE}.ipynb
done
