# !/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

INFEATURES=2
OUTFEATURES=3
HIDDENLAYERS=3
HIDDENFEATURES=256

DSET="Chest_CT"


case $DSET in
    "Chest_CT")
        DATADIR="/local_dataset/Chest_CT" 
        SIDELEN=128
        MAXITER=120
        NUM_EPOCHS=2000
        ISGRAY=1
        ;;
    "DIV2K")
        DATADIR="/local_dataset/DIV2K"
        MAXITER=40
        SIDELEN=128
        NUM_EPOCHS=500       # Pretrain Image epoch
        ISGRAY=0
        ;;
    *)
        echo "Unknown dataset type. Please choose one of: Chest_CT"
        exit 1
        ;;
esac


## For scratch
GPUID_TUPLE=(0 1 2 3 4 5)
TARGET_IDS=(46 92 59 112 100 1 33 70)
declare -A imgid_map
ID=0


for IMGID in "${TARGET_IDS[@]}";            
    do
        LOGDIR="$BASE_DIR/gradio_deploy/results/siren/$DSET/scratch"

        GPUID=${GPUID_TUPLE[$ID]}
        imgid_map[$GPUID]="${imgid_map[$GPUID]}, $IMGID"
        ((ID++))
        ID=$((ID % 6))

        # Siren
        CUDA_VISIBLE_DEVICES=$GPUID python $BASE_DIR/train_image_for_gradio.py --imgid $IMGID --datadir $DATADIR \
            --model_type Siren \
            --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
            --first_omega 30 --hidden_omega 30 \
            --init_method sine \
            --lr 1e-4 --num_epochs $NUM_EPOCHS \
            --side_len $SIDELEN \
            --logdir $LOGDIR &
    done

for GPUID in $(seq 0 $((${#GPUID_TUPLE[@]} - 1))); 
    do
        echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"
    done
    wait



# For trasfer learning
# GPUID_TUPLE=(0 1 2 3 4 5)
# SOURCE_IDS=(5 2 1 27)
# TARGET_IDS=(46 92 59 112 100 1 33 70)
# declare -A imgid_map
# ID=0


# for PRETRAINEPOCH in 500
#     do
#         for SOURCEIMGID in "${SOURCE_IDS[@]}";
#             do
#                 LOADPATH="$BASE_DIR/gradio_deploy/results/siren/DIV2K/scratch/imid[$SOURCEIMGID]_Siren_3x256_init[sine]_fbs[None]_lr[0.0001]_fw[30.0]_hw[30.0].pt"
#                 unset imgid_map
#                 declare -A imgid_map

#                 for TARGETIMGID in "${TARGET_IDS[@]}";
#                     do
#                         LOGDIR="$BASE_DIR/gradio_deploy/results/siren/$DSET/$SOURCEIMGID"

#                         GPUID=${GPUID_TUPLE[$ID]}
#                         imgid_map[$GPUID]="${imgid_map[$GPUID]}, $TARGETIMGID"
#                         ((ID++))
#                         ID=$((ID % 6))

#                         # Siren
#                         CUDA_VISIBLE_DEVICES=$GPUID python $BASE_DIR/train_image_for_gradio.py --imgid $TARGETIMGID --datadir $DATADIR \
#                             --model_type Siren \
#                             --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES --in_features $INFEATURES --out_features $OUTFEATURES \
#                             --first_omega 30 --hidden_omega 30 \
#                             --init_method sine \
#                             --lr 1e-4 --num_epochs $NUM_EPOCHS \
#                             --side_len $SIDELEN \
#                             --load_path $LOADPATH \
#                             --logdir $LOGDIR &
#                     done

#                 echo "SOURCEIMGID: $SOURCEIMGID"
#                 for GPUID in $(seq 0 $((${#GPUID_TUPLE[@]} - 1))); 
#                     do
#                         echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"
#                     done
#                     wait
#             done
#             wait
#     done
#     wait
