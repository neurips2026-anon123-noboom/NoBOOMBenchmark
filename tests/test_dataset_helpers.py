from noboom_benchmark.noboom_lib.core.benchmark_utils import dataset_helpers


class _DummyDataset:
    def __len__(self) -> int:
        return 16


class _FakeConcatDataset:
    def __init__(self, datasets) -> None:
        self._datasets = list(datasets)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)


def test_get_dataloader_sets_worker_init_fn_for_multi_worker_loaders(monkeypatch) -> None:
    captured = {}

    class FakeDataLoader:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_helpers, "PipelineDataset", lambda data: data)
    monkeypatch.setattr(dataset_helpers, "ConcatDataset", _FakeConcatDataset)
    monkeypatch.setattr(dataset_helpers, "collate_fn", lambda *, batch_dim: ("collate", batch_dim))
    monkeypatch.setattr(dataset_helpers.torch.utils.data, "DataLoader", FakeDataLoader)

    dataset_helpers.get_dataloader(
        data=_DummyDataset(),
        pipeline=None,
        batch_dim=0,
        shuffle=False,
        num_workers=2,
    )

    assert captured["kwargs"]["worker_init_fn"] is dataset_helpers._init_dataloader_worker
