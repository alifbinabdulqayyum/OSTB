for num_gaussians in 1; 
do
    for mod_depth in 4;
    do
        python train.py \
            --num-gaussians $num_gaussians \
            --attention 'gaussian-attention' \
            --num-opts 0 \
            --max-opts 120000 \
            --n-query-points 1024 \
            --file-prefix "ua" \
            --batch-size 4 \
            --mod-depth $mod_depth \
            --eval-interval 6000 \
            --save-interval 12000 \
            --max-scale 4.0
    done
done