import pymssql


class DBManager:

    def __init__(self, server='sql16ssd-003.localnet.kr', db='lime425_rmos', user='lime425_rmos', pwd='rmos0425'):
        self._server = server
        self._db = db
        self._user = user
        self._pwd = pwd
        self._conn = pymssql.connect(self._server, self._user, self._pwd, self._db)
