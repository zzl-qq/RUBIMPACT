"""Global key-value store for loose module coupling (framework S9.1)."""
from typing import Any


class DataBus:
    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def keys(self):
        return self._store.keys()
