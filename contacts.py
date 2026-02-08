#!/usr/bin/env python3
"""Query the macOS Apple Contacts database via SQLite."""

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Core Data epoch: 2001-01-01 00:00:00 UTC
CORE_DATA_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

# Apple label encoding pattern: _$!<Label>!$_
LABEL_PATTERN = re.compile(r"_\$!<(.+?)>!\$_")


# ---------------------------------------------------------------------------
# MacOSDatabase — reusable base for macOS SQLite databases
# ---------------------------------------------------------------------------

class MacOSDatabase:
    """Base class for read-only access to macOS SQLite databases."""

    base_dir: str  # e.g. "~/Library/Application Support/AddressBook"
    db_filename: str  # e.g. "AddressBook-v22.abcddb"

    def __init__(self):
        self._conn: sqlite3.Connection | None = None

    # -- path discovery --

    @classmethod
    def find_databases(cls) -> list[Path]:
        """Find all database files under Sources/<UUID>/ subdirectories."""
        base = Path(os.path.expanduser(cls.base_dir)) / "Sources"
        if not base.exists():
            return []
        dbs = []
        for child in sorted(base.iterdir()):
            if child.is_dir() and len(child.name) == 36:  # UUID-length
                db_path = child / cls.db_filename
                if db_path.exists():
                    dbs.append(db_path)
        return dbs

    # -- connection --

    def connect(self, path: Path) -> sqlite3.Connection:
        """Open a read-only connection to the database."""
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        return conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("Database not connected. Call open() first.")
        return self._conn

    # -- utilities --

    @staticmethod
    def apple_timestamp(ts) -> str | None:
        """Convert Core Data timestamp to ISO date string."""
        if ts is None:
            return None
        dt = CORE_DATA_EPOCH + timedelta(seconds=ts)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def decode_label(label: str | None) -> str:
        """Strip Apple's _$!<Label>!$_ encoding to a clean string."""
        if not label:
            return ""
        m = LABEL_PATTERN.match(label)
        return m.group(1) if m else label


# ---------------------------------------------------------------------------
# ContactsDB
# ---------------------------------------------------------------------------

class ContactsDB(MacOSDatabase):
    """Query the macOS Contacts database."""

    base_dir = "~/Library/Application Support/AddressBook"
    db_filename = "AddressBook-v22.abcddb"

    def open(self) -> "ContactsDB":
        """Auto-discover databases and connect to the one with the most contacts."""
        dbs = self.find_databases()
        if not dbs:
            print("Error: No Contacts databases found.", file=sys.stderr)
            sys.exit(1)

        best_path = None
        best_count = -1
        for db_path in dbs:
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                count = conn.execute(
                    "SELECT COUNT(*) FROM ZABCDRECORD WHERE Z_ENT = 22"
                ).fetchone()[0]
                conn.close()
                if count > best_count:
                    best_count = count
                    best_path = db_path
            except sqlite3.Error:
                continue

        if best_path is None:
            print("Error: Could not open any Contacts database.", file=sys.stderr)
            sys.exit(1)

        self.connect(best_path)
        self._db_path = best_path
        return self

    def _format_name(self, row) -> str:
        """Build a display name from first/last/org fields."""
        parts = []
        first = row["ZFIRSTNAME"]
        last = row["ZLASTNAME"]
        if first:
            parts.append(first)
        if last:
            parts.append(last)
        if parts:
            return " ".join(parts)
        org = row["ZORGANIZATION"]
        return org or "(no name)"

    # -- queries --

    def list_contacts(self, limit: int | None = None, sort: str = "first") -> list[dict]:
        """List contacts with name, org, primary phone, and primary email."""
        order = "ZFIRSTNAME, ZLASTNAME" if sort == "first" else "ZLASTNAME, ZFIRSTNAME"
        sql = f"""
            SELECT r.Z_PK, r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION,
                   (SELECT e.ZADDRESS FROM ZABCDEMAILADDRESS e
                    WHERE e.ZOWNER = r.Z_PK ORDER BY e.ZORDERINGINDEX LIMIT 1) AS email,
                   (SELECT p.ZFULLNUMBER FROM ZABCDPHONENUMBER p
                    WHERE p.ZOWNER = r.Z_PK ORDER BY p.ZORDERINGINDEX LIMIT 1) AS phone
            FROM ZABCDRECORD r
            WHERE r.Z_ENT = 22
            ORDER BY {order} COLLATE NOCASE
        """
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = self.conn.execute(sql).fetchall()
        results = []
        for row in rows:
            results.append({
                "pk": row["Z_PK"],
                "name": self._format_name(row),
                "organization": row["ZORGANIZATION"] or "",
                "email": row["email"] or "",
                "phone": row["phone"] or "",
            })
        return results

    def search(self, query: str, field: str = "all") -> list[dict]:
        """Search contacts across multiple fields."""
        like = f"%{query}%"

        conditions = {
            "name": "(r.ZFIRSTNAME LIKE ? OR r.ZLASTNAME LIKE ?)",
            "email": """r.Z_PK IN (
                SELECT e.ZOWNER FROM ZABCDEMAILADDRESS e WHERE e.ZADDRESS LIKE ?)""",
            "phone": """r.Z_PK IN (
                SELECT p.ZOWNER FROM ZABCDPHONENUMBER p WHERE p.ZFULLNUMBER LIKE ?)""",
            "org": "r.ZORGANIZATION LIKE ?",
            "notes": """r.Z_PK IN (
                SELECT n.ZCONTACT FROM ZABCDNOTE n WHERE n.ZTEXT LIKE ?)""",
        }

        if field == "all":
            where_parts = list(conditions.values())
            where = " OR ".join(where_parts)
            # name needs 2 params, everything else needs 1
            params = [like, like, like, like, like, like]
        else:
            where = conditions[field]
            params = [like, like] if field == "name" else [like]

        sql = f"""
            SELECT r.Z_PK, r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION,
                   (SELECT e.ZADDRESS FROM ZABCDEMAILADDRESS e
                    WHERE e.ZOWNER = r.Z_PK ORDER BY e.ZORDERINGINDEX LIMIT 1) AS email,
                   (SELECT p.ZFULLNUMBER FROM ZABCDPHONENUMBER p
                    WHERE p.ZOWNER = r.Z_PK ORDER BY p.ZORDERINGINDEX LIMIT 1) AS phone
            FROM ZABCDRECORD r
            WHERE r.Z_ENT = 22 AND ({where})
            ORDER BY r.ZFIRSTNAME, r.ZLASTNAME COLLATE NOCASE
        """

        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            results.append({
                "pk": row["Z_PK"],
                "name": self._format_name(row),
                "organization": row["ZORGANIZATION"] or "",
                "email": row["email"] or "",
                "phone": row["phone"] or "",
            })
        return results

    def get_contact(self, identifier: str) -> dict | None:
        """Get full contact details by name or PK."""
        # Try as PK first
        if identifier.isdigit():
            row = self.conn.execute(
                "SELECT * FROM ZABCDRECORD WHERE Z_PK = ? AND Z_ENT = 22",
                (int(identifier),),
            ).fetchone()
        else:
            row = None

        # Search by name if not found by PK
        if row is None:
            like = f"%{identifier}%"
            row = self.conn.execute(
                """SELECT * FROM ZABCDRECORD WHERE Z_ENT = 22
                   AND (ZFIRSTNAME || ' ' || ZLASTNAME LIKE ?
                        OR ZFIRSTNAME LIKE ? OR ZLASTNAME LIKE ?
                        OR ZORGANIZATION LIKE ?)
                   LIMIT 1""",
                (like, like, like, like),
            ).fetchone()

        if row is None:
            return None

        pk = row["Z_PK"]
        contact = {
            "pk": pk,
            "first_name": row["ZFIRSTNAME"] or "",
            "last_name": row["ZLASTNAME"] or "",
            "organization": row["ZORGANIZATION"] or "",
            "department": row["ZDEPARTMENT"] or "",
            "job_title": row["ZJOBTITLE"] or "",
            "nickname": row["ZNICKNAME"] or "",
            "birthday": self.apple_timestamp(row["ZBIRTHDAY"]),
            "created": self.apple_timestamp(row["ZCREATIONDATE"]),
            "modified": self.apple_timestamp(row["ZMODIFICATIONDATE"]),
        }

        # Emails
        contact["emails"] = [
            {"label": self.decode_label(r["ZLABEL"]), "address": r["ZADDRESS"]}
            for r in self.conn.execute(
                "SELECT ZLABEL, ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Phones
        contact["phones"] = [
            {"label": self.decode_label(r["ZLABEL"]), "number": r["ZFULLNUMBER"]}
            for r in self.conn.execute(
                "SELECT ZLABEL, ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Addresses
        contact["addresses"] = []
        for r in self.conn.execute(
            """SELECT ZLABEL, ZSTREET, ZCITY, ZSTATE, ZZIPCODE, ZCOUNTRYNAME
               FROM ZABCDPOSTALADDRESS WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX""",
            (pk,),
        ):
            parts = [p for p in [r["ZSTREET"], r["ZCITY"], r["ZSTATE"],
                                  r["ZZIPCODE"], r["ZCOUNTRYNAME"]] if p]
            contact["addresses"].append({
                "label": self.decode_label(r["ZLABEL"]),
                "formatted": ", ".join(parts),
            })

        # URLs
        contact["urls"] = [
            {"label": self.decode_label(r["ZLABEL"]), "url": r["ZURL"]}
            for r in self.conn.execute(
                "SELECT ZLABEL, ZURL FROM ZABCDURLADDRESS WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Social profiles
        contact["social_profiles"] = [
            {"service": r["ZSERVICENAME"] or "", "username": r["ZUSERNAME"] or "",
             "url": r["ZURLSTRING"] or ""}
            for r in self.conn.execute(
                "SELECT ZSERVICENAME, ZUSERNAME, ZURLSTRING FROM ZABCDSOCIALPROFILE WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Related names
        contact["related_names"] = [
            {"label": self.decode_label(r["ZLABEL"]), "name": r["ZNAME"]}
            for r in self.conn.execute(
                "SELECT ZLABEL, ZNAME FROM ZABCDRELATEDNAME WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Dates
        contact["dates"] = [
            {"label": self.decode_label(r["ZLABEL"]), "date": self.apple_timestamp(r["ZDATE"])}
            for r in self.conn.execute(
                "SELECT ZLABEL, ZDATE FROM ZABCDCONTACTDATE WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX",
                (pk,),
            )
        ]

        # Notes (uses ZCONTACT, not ZOWNER)
        note_row = self.conn.execute(
            "SELECT ZTEXT FROM ZABCDNOTE WHERE ZCONTACT = ?", (pk,)
        ).fetchone()
        contact["notes"] = note_row["ZTEXT"] if note_row and note_row["ZTEXT"] else ""

        return contact

    def list_groups(self) -> list[dict]:
        """List all contact groups with member counts."""
        rows = self.conn.execute("""
            SELECT g.Z_PK, g.ZNAME,
                   (SELECT COUNT(*) FROM Z_22PARENTGROUPS pg
                    WHERE pg.Z_19PARENTGROUPS1 = g.Z_PK) AS member_count
            FROM ZABCDRECORD g
            WHERE g.Z_ENT = 19
            ORDER BY g.ZNAME COLLATE NOCASE
        """).fetchall()
        return [{"pk": r["Z_PK"], "name": r["ZNAME"] or "(unnamed)",
                 "members": r["member_count"]} for r in rows]

    def list_sources(self) -> list[dict]:
        """List all source databases with contact counts."""
        sources = []
        for db_path in self.find_databases():
            uuid = db_path.parent.name
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                count = conn.execute(
                    "SELECT COUNT(*) FROM ZABCDRECORD WHERE Z_ENT = 22"
                ).fetchone()[0]
                conn.close()
                sources.append({"uuid": uuid, "contacts": count, "path": str(db_path)})
            except sqlite3.Error as e:
                sources.append({"uuid": uuid, "contacts": 0, "path": str(db_path),
                                "error": str(e)})
        return sources


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class Formatter:
    """Format contact data for terminal or JSON output."""

    @staticmethod
    def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
        """Render rows as a terminal-width-aware aligned table.

        columns: list of (key, header) tuples.
        """
        if not rows:
            return "No results."

        term_width = shutil.get_terminal_size((100, 25)).columns

        # Calculate column widths
        widths = [len(header) for _, header in columns]
        for row in rows:
            for i, (key, _) in enumerate(columns):
                widths[i] = max(widths[i], len(str(row.get(key, ""))))

        # Shrink last column to fit terminal if needed
        total = sum(widths) + (len(columns) - 1) * 3  # 3 chars per separator
        if total > term_width and len(columns) > 1:
            widths[-1] = max(10, term_width - sum(widths[:-1]) - (len(columns) - 1) * 3)

        lines = []
        # Header
        header = " | ".join(h.ljust(widths[i]) for i, (_, h) in enumerate(columns))
        lines.append(header)
        lines.append("-+-".join("-" * w for w in widths))

        # Rows
        for row in rows:
            parts = []
            for i, (key, _) in enumerate(columns):
                val = str(row.get(key, ""))
                if len(val) > widths[i]:
                    val = val[: widths[i] - 1] + "…"
                parts.append(val.ljust(widths[i]))
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def detail(contact: dict) -> str:
        """Format a full contact detail view."""
        lines = []

        # Name header
        name_parts = [contact.get("first_name", ""), contact.get("last_name", "")]
        name = " ".join(p for p in name_parts if p) or "(no name)"
        lines.append(f"  {name}")
        lines.append("  " + "=" * len(name))

        # Basic fields
        fields = [
            ("Organization", "organization"),
            ("Department", "department"),
            ("Job Title", "job_title"),
            ("Nickname", "nickname"),
            ("Birthday", "birthday"),
        ]
        for label, key in fields:
            val = contact.get(key)
            if val:
                lines.append(f"  {label}: {val}")

        # Multi-value sections
        sections = [
            ("Phones", "phones", lambda r: f"{r.get('label', ''):>10}  {r.get('number', '')}"),
            ("Emails", "emails", lambda r: f"{r.get('label', ''):>10}  {r.get('address', '')}"),
            ("Addresses", "addresses", lambda r: f"{r.get('label', ''):>10}  {r.get('formatted', '')}"),
            ("URLs", "urls", lambda r: f"{r.get('label', ''):>10}  {r.get('url', '')}"),
            ("Social", "social_profiles", lambda r: f"{r.get('service', ''):>10}  {r.get('username', '')}"),
            ("Related", "related_names", lambda r: f"{r.get('label', ''):>10}  {r.get('name', '')}"),
            ("Dates", "dates", lambda r: f"{r.get('label', ''):>10}  {r.get('date', '')}"),
        ]

        for title, key, fmt in sections:
            items = contact.get(key, [])
            if items:
                lines.append(f"\n  {title}:")
                for item in items:
                    lines.append(f"    {fmt(item)}")

        # Notes
        notes = contact.get("notes", "")
        if notes:
            lines.append(f"\n  Notes:")
            for line in notes.splitlines():
                lines.append(f"    {line}")

        # Metadata
        lines.append(f"\n  PK: {contact['pk']}")
        if contact.get("created"):
            lines.append(f"  Created: {contact['created']}")
        if contact.get("modified"):
            lines.append(f"  Modified: {contact['modified']}")

        return "\n".join(lines)

    @staticmethod
    def json_output(data) -> str:
        """Render data as formatted JSON."""
        return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(db: ContactsDB, args):
    contacts = db.list_contacts(limit=args.n, sort=args.sort)
    if args.json:
        print(Formatter.json_output(contacts))
    else:
        print(Formatter.table(contacts, [
            ("name", "Name"), ("organization", "Org"),
            ("phone", "Phone"), ("email", "Email"),
        ]))
        print(f"\n{len(contacts)} contacts")


def cmd_search(db: ContactsDB, args):
    results = db.search(args.query, field=args.field)
    if args.json:
        print(Formatter.json_output(results))
    else:
        print(Formatter.table(results, [
            ("name", "Name"), ("organization", "Org"),
            ("phone", "Phone"), ("email", "Email"),
        ]))
        print(f"\n{len(results)} results")


def cmd_show(db: ContactsDB, args):
    identifier = " ".join(args.name_or_pk)
    contact = db.get_contact(identifier)
    if contact is None:
        print(f"No contact found for: {identifier}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(Formatter.json_output(contact))
    else:
        print(Formatter.detail(contact))


def cmd_groups(db: ContactsDB, args):
    groups = db.list_groups()
    if args.json:
        print(Formatter.json_output(groups))
    else:
        print(Formatter.table(groups, [
            ("name", "Group"), ("members", "Members"),
        ]))


def cmd_sources(db: ContactsDB, args):
    sources = db.list_sources()
    if args.json:
        print(Formatter.json_output(sources))
    else:
        print(Formatter.table(sources, [
            ("uuid", "UUID"), ("contacts", "Contacts"), ("path", "Path"),
        ]))


def main():
    parser = argparse.ArgumentParser(
        description="Query macOS Apple Contacts via SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List all contacts")
    p_list.add_argument("-n", type=int, default=None, help="Limit number of results")
    p_list.add_argument("--sort", choices=["first", "last"], default="first",
                        help="Sort by first or last name")
    p_list.add_argument("--json", action="store_true", help="JSON output")
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = sub.add_parser("search", help="Search contacts")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--field", choices=["name", "email", "phone", "org", "notes", "all"],
                          default="all", help="Field to search (default: all)")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.set_defaults(func=cmd_search)

    # show
    p_show = sub.add_parser("show", help="Show full contact details")
    p_show.add_argument("name_or_pk", nargs="+", help="Contact name or PK")
    p_show.add_argument("--json", action="store_true", help="JSON output")
    p_show.set_defaults(func=cmd_show)

    # groups
    p_groups = sub.add_parser("groups", help="List contact groups")
    p_groups.add_argument("--json", action="store_true", help="JSON output")
    p_groups.set_defaults(func=cmd_groups)

    # sources
    p_sources = sub.add_parser("sources", help="List source databases")
    p_sources.add_argument("--json", action="store_true", help="JSON output")
    p_sources.set_defaults(func=cmd_sources)

    args = parser.parse_args()

    with ContactsDB() as db:
        db.open()
        args.func(db, args)


if __name__ == "__main__":
    main()
