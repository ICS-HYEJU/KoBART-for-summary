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
        pretrained_name='gogamza/kobart-base-v1',
        train_file='/storage/hjchoi/Document_Summary_text/Training/law.tsv',
        test_file='/storage/hjchoi/Document_Summary_text/Training/law.tsv',
        batch_size=10,
        max_len=512,
        max_epoch=50,
        num_workers=4,
    )
    path = dict(
        save_base_path='/home/hjchoi/PycharmProjects/KoBART-for-summary/runs',
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
        frequency=1,
    )
    weight_info = dict(
        mode='min',
        checkpoint='./checkpoint/',
        save_fname='model_chp/{epoch:02d}-{val_loss:.3f}',
    )
    device=dict(
        gpu_id=1,
    )
    config=dict(
        dataset_info=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
        weight_info=weight_info,
        device=device
    )
    return Config(config)