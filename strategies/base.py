from __future__ import annotations

from abc import ABC, abstractmethod

if False:
    from warehouse_simulator.map import WarehouseMap


class StorageLocationStrategy(ABC):
    name: str = "Base"

    @abstractmethod
    def assign(
        self,
        sku_list: list[int],
        warehouse_map: "WarehouseMap",
        order_data: list[dict],
    ) -> dict[int, list[int]]:
        """Assign SKUs to lanes and return lane_id -> SKU list."""

