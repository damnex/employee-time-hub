from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class FaceDatabaseConfig:
    path: str


class FaceDatabase:
    def __init__(self, config: FaceDatabaseConfig) -> None:
        self._path = Path(config.path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_payload({"employees": []})

    def list_faces(self) -> list[dict[str, object]]:
        with self._lock:
            payload = self._read_payload()
            return list(payload.get("employees", []))

    def upsert_employee(
        self,
        *,
        employee_id: str,
        name: str,
        rfid_tag: str | None,
        embeddings: list[list[float]],
    ) -> dict[str, object]:
        if not embeddings:
            raise ValueError("At least one embedding is required.")

        with self._lock:
            payload = self._read_payload()
            employees = list(payload.get("employees", []))
            now = datetime.now(UTC).isoformat()
            normalized_tag = rfid_tag.strip().upper() if rfid_tag else None
            new_employee = {
                "id": employee_id.strip(),
                "name": name.strip(),
                "rfid_tag": normalized_tag,
                "embeddings": embeddings,
                "embedding_count": len(embeddings),
                "updated_at": now,
            }

            updated = False
            for index, employee in enumerate(employees):
                if str(employee.get("id")) == new_employee["id"]:
                    employees[index] = new_employee
                    updated = True
                    break
            if not updated:
                employees.append(new_employee)

            payload["employees"] = employees
            self._write_payload(payload)
            return new_employee

    def _read_payload(self) -> dict[str, object]:
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_payload(self, payload: dict[str, object]) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

