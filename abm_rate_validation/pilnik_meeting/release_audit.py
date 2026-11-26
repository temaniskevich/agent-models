"""Даты, к которым документированы входы нового ретроспективного теста."""

import csv
import html
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MONTHS = [f"2025-{m:02d}" for m in range(9, 13)] + [f"2026-{m:02d}" for m in range(1, 10)]
DATE = r"(\d{2}\.\d{2}\.\d{4})"


def download(url):
    result = subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "1",
                             "--connect-timeout", "6", "--max-time", "18", url],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def archive_text(source):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", source)).split())


def official_date(text, pattern, url):
    match = re.search(pattern + DATE, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Дата публикации не найдена: {url}")
    return datetime.strptime(match[1], "%d.%m.%Y").date()


def archived_subsidized(source, month):
    for match in re.finditer(r"var settings\s*=\s*", source):
        chart, _ = json.JSONDecoder().raw_decode(source[match.end():])
        for series in chart.get("series", []):
            if series.get("name", "").strip() == "Льготные ИЖК":
                for point in series["data"]:
                    if datetime.fromtimestamp(point["x"] / 1000, tz=timezone.utc).strftime("%Y-%m") == month:
                        return point["y"]
    return None


def audit(month):
    year, mm = month.split("-")
    stamp = mm + year[-2:]
    base = "https://www.cbr.ru/statistics/bank_sector"
    rates_url = f"{base}/int_rat/{stamp}/"
    mortgage_url = f"{base}/mortgage/Indicator_mortgage/{stamp}/"
    rates, mortgage = download(rates_url), download(mortgage_url)
    rates_date = official_date(archive_text(rates), r"Дата последнего обновления:\s*", rates_url)
    mortgage_date = official_date(archive_text(mortgage), r"дата размещения бюллетеня за .*?\s*[–—-]\s*", mortgage_url)
    subsidiary_rate = archived_subsidized(rates, month)
    if subsidiary_rate is None:
        subsidiary_rate = archived_subsidized(mortgage, month)
    return {"period": month, "available_by": max(rates_date, mortgage_date), "rates_available_by": rates_date,
            "bulletin_available_by": mortgage_date, "archived_subsidized_pct": subsidiary_rate,
            "rates_url": rates_url, "mortgage_url": mortgage_url}


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(audit, MONTHS))
    with (HERE / "release_dates.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
