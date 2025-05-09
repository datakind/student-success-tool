import typing as t


class ModelCardConfig(t.Protocol):
    institution_id: str
    institution_name: str
    modeling: t.Any
