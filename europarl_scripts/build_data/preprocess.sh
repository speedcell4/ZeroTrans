#!/bin/bash

ROOT_PATH="../"

mkdir $ROOT_PATH/europarl_scripts/logs
mkdir $ROOT_PATH/europarl_scripts/checkpoints
mkdir $ROOT_PATH/europarl_scripts/results

# download dataset
curl -o mmcr4nlp.zip https://lotus.kuee.kyoto-u.ac.jp/~raj/mmcr4nlp
unzip mmcr4nlp.zip

# extract mono data
ROW_DATA_PATH=${ROOT_PATH}/europarl_scripts/mmcr4nlp/europarl
MONO_PATH=${ROOT_PATH}/europarl_scripts/build_data/mono
mkdir $MONO_PATH
python ${ROOT_PATH}/europarl_scripts/build_data/mono_reader.py $ROW_DATA_PATH $MONO_PATH

# tokenization
TOKENIZER=${ROOT_PATH}/moses/scripts/tokenizer/tokenizer.perl
TOKENIZED_PATH=${ROOT_PATH}/europarl_scripts/build_data/tokenized
mkdir $TOKENIZED_PATH
for lang in en de fi pt bg sl it pl hu ro es da nl et cs; do
  for split in train valid test; do
    file_name=${MONO_PATH}/${split}"."${lang}
    cat $file_name | perl $TOKENIZER -threads 64 -l $lang >> ${TOKENIZED_PATH}"/"${split}"."$lang
  done
done

# bpe
FAIR_PATH=${ROOT_PATH}/fairseq
SCRIPTS=${FAIR_PATH}/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py
SPM_DECODE=$SCRIPTS/spm_decode.py
BPESIZE=50000
TRAIN_FILES=${ROOT_PATH}/europarl_scripts/build_data/bpe.input-output

# integrate training data for bpe (sentencepiece)
for lang in en de fi pt bg sl it pl hu ro es da nl et cs; do
  filename=${TOKENIZED_PATH}"/train."$lang
  echo $filename
  cat $filename >> $TRAIN_FILES
done

# get bpe model
echo "learning joint BPE over ${TRAIN_FILES}..."
python $SPM_TRAIN \
    --input=$TRAIN_FILES \
    --model_prefix=${ROOT_PATH}/europarl_scripts/build_data/europarl.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode row data via bpe
BPE_MONO_PATH=${ROOT_PATH}/europarl_scripts/build_data/bpe_mono
mkdir $BPE_MONO_PATH
for lang in en de fi pt bg sl it pl hu ro es da nl et cs; do
  for split in train valid test; do
    python "$SPM_ENCODE" \
          --model ${ROOT_PATH}/europarl_scripts/build_data/europarl.bpe.model \
          --output_format=piece \
          --inputs ${TOKENIZED_PATH}"/"$split"."$lang \
          --outputs ${BPE_MONO_PATH}"/"$split"."$lang
  done
done
BPE_PATH=${ROOT_PATH}/europarl_scripts/build_data/bpe
mkdir $BPE_PATH
# pairing monolingual data
python ${ROOT_PATH}/europarl_scripts/build_data/pairing.py $BPE_MONO_PATH $BPE_PATH

# get dict
BINARY_PATH=${ROOT_PATH}/europarl_15-bin
mkdir $BINARY_PATH
cut -f 1 ${ROOT_PATH}/europarl_scripts/build_data/europarl.bpe.vocab | tail -n +4 | sed "s/$/ 1/g" > ${BINARY_PATH}/dict.txt

# binary by fairseq
for src in en de fi pt bg sl it pl hu ro es da nl et cs; do
  for tgt in en de fi pt bg sl it pl hu ro es da nl et cs; do
    if [ $src == $tgt ]; then
      continue
    fi
    if [ $src == 'en' ] || [ $tgt == 'en' ]; then
        fairseq-preprocess --task "translation" --source-lang $src --target-lang $tgt \
        --trainpref ${BPE_PATH}/train.${src}"_"${tgt} \
        --validpref ${BPE_PATH}/valid.${src}"_"${tgt} \
        --testpref ${BPE_PATH}/test.${src}"_"${tgt} \
        --destdir ${BINARY_PATH} --padding-factor 1 --workers 128 \
        --srcdict ${BINARY_PATH}/dict.txt --tgtdict ${BINARY_PATH}/dict.txt
    fi
    if [ $src != 'en' ] && [ $tgt != 'en' ]; then
        fairseq-preprocess --task "translation" --source-lang $src --target-lang $tgt \
        --testpref ${BPE_PATH}/test.${src}"_"${tgt} \
        --destdir ${BINARY_PATH} --padding-factor 1 --workers 128 \
        --srcdict ${BINARY_PATH}/dict.txt --tgtdict ${BINARY_PATH}/dict.txt
    fi
  done
done
