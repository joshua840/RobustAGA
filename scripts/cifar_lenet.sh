OPTIMIZER_ARGS='--optimizer adam --weight_decay 4e-5 --milestones 100 150'
DATA_ARGS='--dataset cifar10 --data_dir /data/cifar10'
TRAINER_ARGS='--max_epochs 300 --accelerator gpu --devices 1'
DEFAULT_ARGS='--model lenet --loggername tensorboard --default_root_dir ./output/YOUR_PROJECT'

# Without regularization
python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS


# With hessian
python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS \
--regularizer hessian  --lamb 1e-3


# With l2 + cosd
python project/main.py $OPTIMIZER_ARGS $DATA_ARGS $TRAINER_ARGS $DEFAULT_ARGS \
--regularizer l2_cosd --eps 4 --lamb_l2 0.01 --lamb_cos 1.0 