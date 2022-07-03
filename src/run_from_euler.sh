#BSUB -W 48:00
#BSUB -n 1
#BSUB -o euleroutputs/outfile_%J.%I.txt
#BSUB -R "rusage[mem=15000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=7500]"
#BSUB -R "rusage[scratch=55000]"
#BSUB -J "BB1_bs16"

echo "starting script from sh file"

# echo "copying data to tmpdir..."
# mkdir $TMPDIR/datasets
# mkdir $TMPDIR/datasets/dataset_downsampled
# cp -r /cluster/scratch/jbaumer/datasets/ $TMPDIR
# echo "done copying data to tmpdir"

echo TEMPDIR: $TMPDIR

# activate Enviroment
source ../venv/bin/activate
echo "environment activated"

# load modules
# module load gcc/8.2.0 gdal/3.2.0 zlib/1.2.9 eth_proxy hdf5/1.10.1
module load eth_proxy
echo "modules loaded"

echo "starting actual python script..."
python model_fog_classification.py  --batch_size 16 --lr 0.0001 --epochs 100 --train_split 0.8 --path_dset /cluster/scratch/jbaumer/datasets/dataset_downsampled/ --weighted Auto --model resnet --optim SGD --lr_scheduler 15 --weight_decay 0.1 --momentum 0.9  --stations "['Buelenberg_1']"
