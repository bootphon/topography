for dataset in birddcase cifar speechcommands
do
    for seed in 1 2 3 4
    do
        python $dataset/create_jobs.py -w $WORK --seed $seed
    done
done