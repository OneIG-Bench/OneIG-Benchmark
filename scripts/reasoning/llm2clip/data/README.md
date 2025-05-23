## Train
   ```bash
   $DATASET=cc3m #options: "cc3m", "cc12m", "yfcc15m"
   bash download_dataset.sh $DATASET
   python extract_embedding.py $DATASET
   ```
## Eval
   ```bash
   bash setup_eval_datasets.sh
   python extract_eval_embedding.py
   ```