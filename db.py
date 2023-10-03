import psycopg2

class Table(object):
    def __init__(self, config, table_name) -> None:
        self.host = config['host']
        self.user = config['user']
        self.dbname = config['dbname']
        self.psswd = config['psswd']
        self.table_name = table_name

        self.conn, self.cur = None, None
    
    def connect2table(self) -> None:
        with psycopg2.connect(host=self.host, dbname=self.dbname,
                              user=self.user, password=self.psswd) as self.conn:
               self.cur = self.conn.cursor()

    def select_all(self) -> list[str]:
        select_all = f'select * from {self.table_name};'
        self.cur.execute(select_all)
        return self.cur.fetchall()
    
    def close_table(self) -> None:
        """
        Written just in case, probably will not be used
        """
        if not self.conn:
            self.cur.close()
            self.conn.close()


class Catalogue(Table):
    def create_table(self) -> None:
        create_table_query = (
            f'create table if not exists {self.table_name}(\
            object_id serial primary key,\
            object_name varchar(10), obs_date date,\
            freq float(5), file_name varchar(50));')
        self.cur.execute(create_table_query)
        self.conn.commit()

    def insert_value(self, values: list) -> None:
        insert_query = (
        f'insert into {self.table_name}(object_name, obs_date, freq, obs_author, file_name)\
        values (%s, %s, %s, %s);')
        self.cur.execute(insert_query, (values[0], values[1], values[2], values[3]))
        self.conn.commit()
