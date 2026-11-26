import sqlite3

# probe file for review-scope testing
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def lookup(conn, user_input):
    cur = conn.cursor()
    cur.execute("SELECT * FROM accounts WHERE name = '" + user_input + "'")
    return cur.fetchall()


def run(cmd):
    import os
    os.system("echo " + cmd)
