Для обучения ввести эту команду:
python train.py --model unet --backbone resnet34 --batch_size 16 --num_epochs 150 --lr 1e-3 --tmax 150 --experiment_name unet_resnet_jpg_30k_multiclass

Внутри кода поставил в train.py возможность сохранять checkpoints. 
