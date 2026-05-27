import functools
import logging
import time

logger = logging.getLogger("jarvis")


def send_notification(title: str, message: str) -> None:
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message[:200],
            app_name="JARVIS",
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Desktop notification unavailable: %s — %s: %s", title, message, exc)


def with_retry(max_attempts: int = 3, agent_name: str = "Agent"):
    """Decorator: retry up to max_attempts, send desktop notification on final failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning("%s attempt %d/%d failed: %s", agent_name, attempt + 1, max_attempts, exc)
                    if attempt < max_attempts - 1:
                        time.sleep(2 ** attempt)
            send_notification(f"{agent_name} Failed", str(last_exc)[:150])
            raise last_exc
        return wrapper
    return decorator


def agent_status_entry(status: str, result_preview: str = "") -> dict:
    from datetime import datetime
    return {
        "status": status,
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "preview": result_preview[:120],
    }
