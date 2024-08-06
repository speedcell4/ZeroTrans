#!/usr/bin/env bash

METHOD=$1
ID=$2

ROOT_PATH="../"
cd ${ROOT_PATH}/fairseq

# average top-5 checkpoints
checkpoints=$(ls ${ROOT_PATH}/europarl_scripts/checkpoints/${METHOD}/$ID/checkpoint.best_loss_* | tr '\n' ' ')
python3 ${ROOT_PATH}/fairseq/scripts/average_checkpoints.py \
  --inputs $checkpoints \
  --output ${ROOT_PATH}/europarl_scripts/checkpoints/${METHOD}/${METHOD}/$ID/checkpoint_averaged.pt

DIR=""
TASK='translation_multi_simple_epoch'
if [ ${METHOD} == 'vanilla' ]; then
  echo ${METHOD}
  echo ${DIR}
  echo ${TASK}
fi
if [ ${METHOD} == 'zero' ]; then
  DIR="--user-dir models/ZeroTrans "
  TASK='translation_multi_simple_epoch_zero'
  echo ${METHOD}
  echo ${DIR}
  echo ${TASK}
fi

mkdir ${ROOT_PATH}/results/${METHOD}/${ID}

for src in en de fi pt bg sl it pl hu ro es da nl et cs; do
  for tgt in en de fi pt bg sl it pl hu ro es da nl et cs; do
    if [[ $src != $tgt ]]; then
      tgt_file=$src"-"$tgt".raw.txt"
      CUDA_VISIBLE_DEVICES=0, fairseq-generate $ROOT_PATH/europarl_15-bin/ --gen-subset test \
        $DIR \
        -s $src -t $tgt \
        --langs "en,de,nl,da,es,pt,ro,it,sl,pl,cs,bg,fi,hu,et" \
        --lang-pairs "de-en,en-de,nl-en,en-nl,da-en,en-da,es-en,en-es,pt-en,en-pt,ro-en,en-ro,it-en,en-it,sl-en,en-sl,pl-en,en-pl,cs-en,en-cs,bg-en,en-bg,fi-en,en-fi,hu-en,en-hu,et-en,en-et" \
        --path ${ROOT_PATH}/europarl_scripts/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt \
        --remove-bpe sentencepiece \
        --task ${TASK} \
        --encoder-langtok tgt \
        --beam 4 >${ROOT_PATH}/europarl_scripts/results/${METHOD}/${ID}/$tgt_file

      # hypothesis
      cat ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- >${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".h"
      # reference
      cat ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- >${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".r"
      rm ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$tgt_file

      cat ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".h" | perl ${ROOT_PATH}/moses/scripts/tokenizer/detokenizer.perl -threads 32 -l $tgt >>${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.h"
      cat ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".r" | perl ${ROOT_PATH}/moses/scripts/tokenizer/detokenizer.perl -threads 32 -l $tgt >>${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.r"
      rm ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".h"
      rm ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".r"

      echo $src"-"$tgt >>${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$ID".sacrebleu"
      sacrebleu ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.h" -w 4 -tok 13a <${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.r" >>${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$ID".sacrebleu"
      rm ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.h"
      rm ${ROOT_PATH}/europarl_scripts/results/${METHOD}/$ID/$src"-"$tgt".detok.r"
    fi
  done
done

python ${ROOT_PATH}/europarl_scripts/evaluation/europarl_bertscore.py ${METHOD} ${ID} 0
