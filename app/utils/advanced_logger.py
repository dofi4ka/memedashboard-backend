import logging
from collections import ChainMap
from contextlib import contextmanager
from typing import Any, Dict, Optional


class AdvancedLogger:
    """
    A logger that maintains context via a ChainMap
    for merged key-value items when logging.
    """

    def __init__(self, name: str, **items: Any):
        self.name: str = name
        self._logger: logging.Logger = logging.getLogger(name)
        self._items: ChainMap[str, Any] = ChainMap(items)

    def _send_message(
        self, level: int, message: str, message_items: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, self._get_message(message, message_items))

    def _get_message(
        self, message: str = "", message_items: Optional[Dict[str, Any]] = None
    ) -> str:
        return " ".join(
            f"{k}={v!r}"
            for k, v in self.get_unique_ordered_items(
                {"msg": message}, message_items or {}, *self._items.maps
            )
        )

    @staticmethod
    def get_unique_ordered_items(*maps):
        """Merge dictionaries, keeping the first occurrence of each key."""
        seen = set()
        result = []
        for dictionary in maps:
            for key, value in dictionary.items():
                if key not in seen:
                    seen.add(key)
                    result.append((key, value))
        return result

    @contextmanager
    def scope(self, **items):
        """Temporarily add items to the logging context."""
        self._items.maps.insert(0, items)
        try:
            yield
        finally:
            self._items.maps.pop(0)

    def debug(self, message: str, **items: Any) -> None:
        self._send_message(logging.DEBUG, message, items)

    def info(self, message: str, **items: Any) -> None:
        self._send_message(logging.INFO, message, items)

    def warning(self, message: str, **items: Any) -> None:
        self._send_message(logging.WARNING, message, items)

    def error(self, message: str, **items: Any) -> None:
        self._send_message(logging.ERROR, message, items)

    def critical(self, message: str, **items: Any) -> None:
        self._send_message(logging.CRITICAL, message, items)

    def update(self, **items: Any) -> None:
        self._items.maps[-1].update(items)

    def clear(self) -> None:
        self._items.maps[-1].clear()
