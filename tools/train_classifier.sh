python tools/train_classifier.py \
    --data SAHU \
    --root_path ./dataset/SAHU \
    --seq_len 72 \  # 确保与图像大小匹配
    --step 72 \      # 确保与图像大小匹配
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num_classes 3 \
    --embed_dims 32 \
    --num_stages 3 \
    --device cuda