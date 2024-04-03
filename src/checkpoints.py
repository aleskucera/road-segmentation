from pytorch_lightning.callbacks import ModelCheckpoint

# Instantiate the ModelCheckpoint callback
val_checkpoint = ModelCheckpoint(
    monitor='val_jaccard',
    dirpath='checkpoints/',
    filename='{epoch:02d}-{val_jaccard:.2f}',
    # auto_insert_metric_name=False,
    save_last=True,
    mode='max',
)

regular_checkpoint = ModelCheckpoint(
    monitor='epoch',
    dirpath='checkpoints/',
    filename='latest-{epoch:02d}',
    # auto_insert_metric_name=False,
    mode='max',
    every_n_epochs=1,
    save_last=True,
)
