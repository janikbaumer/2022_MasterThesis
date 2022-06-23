!/bin/bash

#BSUB -W 48:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=55000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=7500]"
#BSUB -R "rusage[scratch=55000]"
#BSUB -J "runname"

#cp -r /cluster/work/igp_psr/jbaumer/stylebias/data/OfficeHome $TMPDIR/
cp -r /cluster/scratch/jbaumer/datasets/ $TMPDIR

echo TEMPDIR: $TMPDIR

# activate Enviroment
source ../venv/bin/activate

# load modules
module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1

python model_fog_classification.py  --batch_size 8 --lr 0.0001 --epochs 5 --train_split 0.8 --path_dset $TMPDIR/dataset_downsampled/ --stations \[\'Stillberg_1\'\] --weighted Manual --model resnet --optim SGD --lr_scheduler 15 --weight_decay 0.1 --momentum 0.9
