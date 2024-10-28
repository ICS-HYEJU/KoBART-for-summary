import json

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

def get_config_dict():
    dataset_info = dict(
        pretrained_name='digit82/kobart-summarization', # ainize/kobert-news
        train_file='/storage/hjchoi/Document_Summary_text/Training/nm.tsv',
        test_file='/storage/hjchoi/Document_Summary_text/Validation/nm.tsv',
        batch_size=2,
        max_len=512,
        max_epoch=50,
        num_workers=5,
    )
    model = dict(
        name='BART',
    )
    solver = dict(
        name='AdamW',
        lr=3e-4,
        gradient_clip_val=1.0
    )
    scheduler = dict(
        monitor='loss',
        interval='step',
        accelerator='gpu',
        frequency=1
    )
    weight_info = dict(
        mode='min',
        checkpoint='./checkpoint/',
        save_acc = 'model_chp/acc/news/{epoch:02d}-{val_acc:.3f}',
        save_val='model_chp/news/{epoch:02d}-{val_loss:.3f}',
        save_train='model_chp/news/{epoch:02d}-{train_loss:.3f}',
        # chp_path='/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/last-v2.ckpt/'
    )
    device=dict(
        gpu_id=1,
    )
    config=dict(
        dataset_info=dataset_info,
        model=model,
        solver=solver,
        scheduler=scheduler,
        weight_info=weight_info,
        device=device
    )
    return Config(config)