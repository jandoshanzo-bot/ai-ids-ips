"""
Server-Sent Events (SSE) модулі — нақты уақыт хабарламалары
=====================================================================
Не істейді:
  Браузерге HTTP арқылы үздіксіз деректер жіберу.
  WebSocket-тен айырмашылығы: бір бағытты (сервер → браузер),
  қарапайым, Flask-те кітапхана қажет емес.

SSE форматы (RFC 6202):
  data: {"prediction": "Attack", "ip": "1.2.3.4"}\n\n

Қолданылған:
  - queue.Queue     — ағындар аралық хабарлама жіберу
  - threading.Lock  — бір уақытта бірнеше клиент қосылса қауіпсіз жұмыс
  - json            — Python dict → JSON мәтін
  - time            — heartbeat (байланыс тірі екенін тексеру)
"""

from __future__ import annotations

import json
import time
import queue
import threading
import logging
from typing import Iterator

logger = logging.getLogger(__name__)


class SSEBroker:
    """
    Барлық қосылған браузер клиенттеріне хабарлама жіберетін брокер.

    Архитектура:
      LogWatcher → SSEBroker.publish() → Queue × N клиент → EventStream → Браузер

    Неліктен Queue?
      Әр браузер-клиентке жеке Queue беріледі. LogWatcher жаңа нәтиже
      жариялағанда (publish), барлық Queue-ларға бір уақытта жазылады.
      Осылайша клиенттер бірін-бірі бөгемейді.
    """

    def __init__(self, maxsize: int = 50):
        """
        Args:
            maxsize: Әр клиент Queue-сының максималды өлшемі.
                     Клиент баяу оқыса — ескі хабарламалар тасталады.
        """
        self._clients: dict[int, queue.Queue] = {}
        self._lock    = threading.Lock()
        self._maxsize = maxsize
        self._counter = 0   # Клиент ID генераторы

    def subscribe(self) -> tuple[int, queue.Queue]:
        """
        Жаңа браузер клиентін тіркейді.
        Returns: (client_id, queue)
        """
        with self._lock:
            self._counter += 1
            cid = self._counter
            q   = queue.Queue(maxsize=self._maxsize)
            self._clients[cid] = q
            logger.debug("SSE клиент қосылды: #%d (барлығы: %d)", cid, len(self._clients))
        return cid, q

    def unsubscribe(self, client_id: int):
        """Клиент байланысын үзгенде тіркемен өшіру."""
        with self._lock:
            self._clients.pop(client_id, None)
            logger.debug("SSE клиент кетті: #%d (қалды: %d)", client_id, len(self._clients))

    def publish(self, data: dict):
        """
        Барлық қосылған клиенттерге хабарлама жіберу.

        Args:
            data: JSON-ға айналдырылатын сөздік
        """
        payload = json.dumps(data, ensure_ascii=False, default=str)
        dead: list[int] = []

        with self._lock:
            clients_snapshot = dict(self._clients)

        for cid, q in clients_snapshot.items():
            try:
                # nowait: Queue толы болса — ескі хабарламаны тастап, жаңасын жаз
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(payload)
            except Exception:
                dead.append(cid)

        # Ажыраған клиенттерді тазалау
        if dead:
            with self._lock:
                for cid in dead:
                    self._clients.pop(cid, None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


def event_stream(broker: SSEBroker, client_id: int, q: queue.Queue) -> Iterator[str]:
    """
    Flask route-тен қайтарылатын generator функция.

    SSE форматы:
      "data: {json}\n\n"  — нақты хабарлама
      ": heartbeat\n\n"   — байланыс тірі екенін тексеру (30 сек сайын)

    Args:
        broker:    SSEBroker нысаны
        client_id: Клиент идентификаторы
        q:         Осы клиентке тән Queue
    """
    try:
        # Қосылу хабарламасы
        yield f"data: {json.dumps({'type': 'connected', 'client_id': client_id})}\n\n"

        last_heartbeat = time.time()

        while True:
            try:
                # 5 секунд күту, жаңа хабарлама болса — дереу жіберу
                payload = q.get(timeout=5)
                yield f"data: {payload}\n\n"

            except queue.Empty:
                # Хабарлама жоқ — heartbeat жіберу (30 сек сайын)
                now = time.time()
                if now - last_heartbeat > 30:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now

    except GeneratorExit:
        # Браузер байланысты үзді
        pass
    finally:
        broker.unsubscribe(client_id)


# ── Глобалды брокер нысаны ─────────────────────────────────────────────────
broker = SSEBroker()

