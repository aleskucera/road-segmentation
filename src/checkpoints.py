from pytorch_lightning.callbacks import ModelCheckpoint

# Instantiate the ModelCheckpoint callback
val_checkpoint = ModelCheckpoint(
    monitor='val_jaccard',
    dirpath='checkpoints/',
    filename='e{epoch:02d}-iou{val_jaccard:.2f}',
    save_last=True,
    mode='min',
)

regular_checkpoint = ModelCheckpoint(
    monitor='epoch',
    dirpath='checkpoints/',
    filename='latest-e{epoch:02d}',
    mode='max',
    every_n_epochs=1,
    save_last=True,
)
