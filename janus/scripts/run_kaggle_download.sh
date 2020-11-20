source scripts/run_setup.sh
source scripts/utils.sh

COMPETITIONS="forest-cover-type-prediction"
COMPETITIONS+=" ghouls-goblins-and-ghosts-boo"
COMPETITIONS+=" otto-group-product-classification-challenge"

MAX_NUM=100

for comp in ${COMPETITIONS}
do
  echo "Downloading Kaggle scripts for ${comp}"

  mkdir -p "${KAGGLE_FOLDER}/${comp}"

  python -m janus.kaggle.download_scripts \
    --competition ${comp} \
    --max_num ${MAX_NUM} \
    --output_dir "${KAGGLE_FOLDER}/${comp}"
done
