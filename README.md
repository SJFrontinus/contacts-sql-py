# contacts-sql-py

Query your macOS Apple Contacts database directly via SQLite. Zero dependencies — just Python's built-in `sqlite3`.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SJFrontinus/contacts-sql-py.git
cd contacts-sql-py
```

No install step needed — `uv run` handles everything.

## Usage

### List contacts

```bash
uv run contacts.py list              # all contacts
uv run contacts.py list -n 20        # first 20
uv run contacts.py list --sort last  # sort by last name
```

### Search

```bash
uv run contacts.py search "Steve"                  # search all fields
uv run contacts.py search "gmail" --field email     # search specific field
uv run contacts.py search "Renown" --field org      # by organization
```

Available fields: `name`, `email`, `phone`, `org`, `notes`, `all` (default)

### Show contact details

```bash
uv run contacts.py show "Jane Smith"   # by name (partial match)
uv run contacts.py show 1047           # by database PK
```

### Groups and sources

```bash
uv run contacts.py groups    # list contact groups with member counts
uv run contacts.py sources   # list all source databases with contact counts
```

### JSON output

Every command supports `--json` for scripting:

```bash
uv run contacts.py search "Steve" --json | python3 -m json.tool
uv run contacts.py show "Jane Smith" --json | jq '.emails'
```

## How it works

The tool reads the Contacts database at `~/Library/Application Support/AddressBook/Sources/<UUID>/AddressBook-v22.abcddb`. macOS may have multiple source databases (iCloud, Exchange, local, etc.) — the tool auto-discovers all of them and connects to the one with the most contacts.

All access is read-only (`?mode=ro`). No Full Disk Access required.

## Architecture

The code is a single file (`contacts.py`) with a reusable base class designed for extending to other macOS databases:

- **`MacOSDatabase`** — base class handling path discovery, read-only SQLite connections, Apple timestamp conversion (Core Data epoch), and label decoding (`_$!<Home>!$_` → `Home`)
- **`ContactsDB(MacOSDatabase)`** — contacts-specific queries across `ZABCDRECORD` and its detail tables (emails, phones, addresses, social profiles, URLs, dates, notes, related names)
- **`Formatter`** — terminal-width-aware table output and JSON serialization
