"""Command line for the catalogue.

    python -m catalog build [--db PATH] [--no-parquet]
    python -m catalog checks [--all]
    python -m catalog tables
    python -m catalog query "SELECT * FROM mart.corpus_summary"
    python -m catalog query --file catalog/queries/01_data_funnel.sql
    python -m catalog docs             # dbt documentation site with the lineage graph

``build`` exits with status 1 when an error-severity check fails, so it can gate a
pipeline or CI job; the catalogue is still written, since it is what you inspect next.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import duckdb

import config

try:
    from logging_setup import setup_logging
except ImportError:  # pragma: no cover - the module is part of the project
    setup_logging = None


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        sys.exit(f"no catalogue at {db_path}; run `python -m catalog build` first")
    return duckdb.connect(db_path, read_only=True)


def _print_checks(con, show_all: bool) -> int:
    rows = con.execute("""
        SELECT check_name, severity, failing_rows, passed, description
        FROM qa.check_results
        ORDER BY passed, severity, check_name""").fetchall()
    failed_errors = 0
    for name, severity, failing, passed, description in rows:
        if passed and not show_all:
            continue
        mark = "PASS" if passed else ("FAIL" if severity == "error" else "WARN")
        failed_errors += (not passed and severity == "error")
        print(f"  {mark:4}  {name:40} {failing:>6} row(s)  {description}")
    n_pass = sum(1 for r in rows if r[3])
    print(f"\n{n_pass}/{len(rows)} checks pass, {failed_errors} error(s). "
          "Offending rows: SELECT * FROM qa.<check_name>")
    return failed_errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m catalog", description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=config.CATALOG_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="rebuild the catalogue from the files on disk")
    p_build.add_argument("--no-parquet", action="store_true",
                         help="skip the Parquet export of the mart tables")

    p_checks = sub.add_parser("checks", help="data-quality report of the last build")
    p_checks.add_argument("--all", action="store_true", help="list passing checks too")

    sub.add_parser("tables", help="list tables and views with row counts")

    sub.add_parser("docs", help="generate the dbt documentation site (lineage graph)")

    p_query = sub.add_parser("query", help="run read-only SQL against the catalogue")
    p_query.add_argument("sql", nargs="?")
    p_query.add_argument("--file")

    args = parser.parse_args(argv)
    # DuckDB draws result tables with box-drawing characters, which a Windows console
    # in cp1252 cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if setup_logging:
        setup_logging()
    else:  # pragma: no cover
        logging.basicConfig(level=logging.INFO)

    if args.command == "build":
        from catalog.build import build

        result = build(db_path=args.db,
                       parquet_dir=None if args.no_parquet else config.CATALOG_PARQUET_DIR)
        print(f"catalogue written to {os.path.relpath(result.db_path, config.ROOT)}\n")
        with duckdb.connect(result.db_path, read_only=True) as con:
            con.sql("SELECT * FROM mart.corpus_summary").show()
            return 1 if _print_checks(con, show_all=False) else 0

    if args.command == "docs":
        from catalog.build import DBT_PROJECT_DIR, generate_docs

        target = generate_docs(db_path=args.db)
        rel = os.path.relpath
        print(f"dbt docs written to {rel(target, config.ROOT)}. To browse them:\n"
              f"  CATALOG_DB_PATH={rel(args.db, config.ROOT)} dbt docs serve "
              f"--project-dir {rel(DBT_PROJECT_DIR, config.ROOT)} "
              f"--profiles-dir {rel(DBT_PROJECT_DIR, config.ROOT)} "
              f"--target-path {rel(target, config.ROOT)}")
        return 0

    con = _connect(args.db)
    try:
        if args.command == "checks":
            return 1 if _print_checks(con, show_all=args.all) else 0
        if args.command == "tables":
            con.sql("""
                SELECT table_schema AS schema, table_name AS name,
                       CASE table_type WHEN 'VIEW' THEN 'view' ELSE 'table' END AS kind
                FROM information_schema.tables
                WHERE table_schema IN ('raw', 'stg', 'mart', 'qa', 'main')
                ORDER BY 1, 2""").show(max_rows=200)
            return 0
        if args.command == "query":
            if bool(args.sql) == bool(args.file):
                parser.error("query takes either SQL text or --file, not both or neither")
            sql = args.sql
            if args.file:
                with open(args.file, encoding="utf-8") as fh:
                    sql = fh.read()
            con.sql(sql).show(max_rows=100, max_width=200)
            return 0
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
