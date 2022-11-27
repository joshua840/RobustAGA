OPTIMIZER_ARGS='--optimizer adamw --weight_decay 1e-2 --milestones 40 60 80'
DATA_ARGS='--dataset imagenet100 --data_dir ~/Data/ImageNet100'
TRAINER_ARGS='--max_epochs 90 --accelerator gpu --devices 1'
DEFAULT_ARGS='--model resnet18_imagenet100 --loggername tensorboard --default_root_dir ./output/YOUR_PROJECT'

# # Without regularization
# python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS


# # With hessian
# python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS \
# --regularizer hessian --lamb 1e-3 --batch_size_train 64


# With l2 + cosd
python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS \
--regularizer l2_cosd --eps 4 --lamb_l2 1.0 --lamb_cos 0.01