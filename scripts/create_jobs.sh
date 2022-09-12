for dataset in birddcase cifar speechcommands
do
    for seed in 0 1 2 3 4
    do
        python $dataset/create_jobs.py -w $WORK --seed $seed
    done
done

cat birddcase*.txt cifar*.txt speechcommands*.txt > all.txt
rm birddcase*.txt cifar*.txt speechcommands*.txt