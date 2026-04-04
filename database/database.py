from peewee import SqliteDatabase

# 1. Initialize the local file-based database
db = SqliteDatabase('atelier_history.db', pragmas={
    'journal_mode': 'wal',  # Write-Ahead Logging for better concurrent performance
    'cache_size': -1024 * 64 # 64MB page cache
})
