for num_gaussians in 1;
do
    # for train_region in 'B';
    # do
    for test_region in 'A' 'B' 'C' 'D';
    do
        python test.py \
            --num-gaussians $num_gaussians \
            --attention 'gaussian-attention' \
            --num-opts 120000 \
            --train-region $train_region \
            --test-region $test_region
    done
    # done
done