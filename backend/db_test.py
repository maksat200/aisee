import sqlite3 

conn = sqlite3.connect("aisee.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM users WHERE email in ('user@example.com', 'admin@example.com')")
conn.commit()

print("DONE")