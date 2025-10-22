# !/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

INFEATURES=2
OUTFEATURES=3
HIDDENLAYERS=3
HIDDENFEATURES=256

DSET="DIV2K"
SIDELEN=256
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dset)
            DSET="$2"
            shift 2
            ;;
        --sidelen)
            SIDELEN="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done


case $DSET in
    "Chest_CT")
        DATADIR="/local_dataset/Chest_CT" 
        SIDELEN=128
        MAXITER=120
        NUM_EPOCHS=2000
        ISGRAY=1
        ;;
    "CelebA_HQ")
        DATADIR="/local_dataset/CelebA_HQ"
        SIDELEN=256
        MAXITER=100
        NUM_EPOCHS=2000
        ISGRAY=0
        ;;
    "DIV2K")
        DATADIR="/local_dataset/DIV2K"
        MAXITER=40
        NUM_EPOCHS=500       # Pretrain Image epoch
        ISGRAY=0
        ;;
    *)
        echo "Unknown dataset type. Please choose one of: Chest_CT, CelebA_HQ, DIV2K"
        exit 1
        ;;
esac



GPUID_TUPLE=(0 1 2 3 4 5 6 7)
declare -A imgid_map
ID=0


## For scratch
# for IMGID in `seq 1 $MAXITER`                
#     do
#         LOGDIR="$BASE_DIR/logs/wire/$DSET/sidelen_$SIDELEN/scratch"

#         GPUID=${GPUID_TUPLE[$ID]}
#         imgid_map[$GPUID]="${imgid_map[$GPUID]}, $IMGID"
#         ((ID++))
#         ID=$((ID % 8))

#         # Wire
#         CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $IMGID --datadir $DATADIR \
#             --model_type Wire \
#             --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
#             --scale 10 --omega_w 20 \
#             --init_method pytorch \
#             --lr 5e-3 --num_epochs $NUM_EPOCHS \
#             --side_len $SIDELEN \
#             --logdir $LOGDIR &
#     done

# for GPUID in $(seq 0 $((${#GPUID_TUPLE[@]} - 1))); do
#     echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"
# done
# wait




# For trasfer learning
for PRETRAINEPOCH in 500
    do
        for SOURCEIMGID in `seq 1 40`
            do
                LOADPATH="$BASE_DIR/logs/wire/DIV2K/sidelen_$SIDELEN/scratch/imid[$SOURCEIMGID]_Wire_3x256_init[pytorch]_fbs[None]_lr[0.005]_omega[5]_scale[10.0]_omegaw[20.0].pt"
                unset imgid_map
                declare -A imgid_map

                for TARGETIMGID in `seq 1 $MAXITER`
                    do
                        LOGDIR="$BASE_DIR/logs/wire/$DSET/sidelen_$SIDELEN/$SOURCEIMGID/pretrain_${PRETRAINEPOCH}step"

                        GPUID=${GPUID_TUPLE[$ID]}
                        imgid_map[$GPUID]="${imgid_map[$GPUID]}, $TARGETIMGID"
                        ((ID++))
                        ID=$((ID % 8))

                        # Wire
                        CUDA_VISIBLE_DEVICES=$GPUID python train_image.py --imgid $TARGETIMGID --datadir $DATADIR \
                            --model_type Wire \
                            --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
                            --scale 10 --omega_w 20 \
                            --init_method pytorch \
                            --lr 5e-3 --num_epochs $NUM_EPOCHS --pretrain_epoch $PRETRAINEPOCH \
                            --side_len $SIDELEN \
                            --load_path $LOADPATH \
                            --logdir $LOGDIR &
                    done

                echo "SOURCEIMGID: $SOURCEIMGID"
                for GPUID in $(seq 0 $((${#GPUID_TUPLE[@]} - 1))); 
                    do
                        echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"
                    done
                    wait
            done
            wait
    done
    wait