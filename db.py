import psycopg2
import yaml
from image import Image
import os


def insert_value(values: list) -> None:
    conn = None
    create_astrogeo = 'create table if not exists astrogeo(\
        object_id serial primary key,\
        object_name varchar(10), obs_date date,\
        freq float(5), file_name varchar(50));'
    insert = (
        "insert into astrogeo(object_name, obs_date, freq, file_name) values (%s, %s, %s, %s);")
    select_all = 'select * from astrogeo;'

    try:
        conn = psycopg2.connect(host=config['host'], dbname=config['dbname'],
                                user=config['user'], password=config['psswd'])
        cur = conn.cursor()

        cur.execute(create_astrogeo)
        cur.execute(insert, (values[0], values[1], values[2], values[3]))
        cur.execute(select_all)
        print(cur.fetchall())
        conn.commit()
        cur.close()

    finally:
        if conn is not None:
            conn.close()


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['db']

# insering 1 object into database
im = Image()
obj = im.get_objects()[0]

path = im.data_path
all_files = os.listdir(f'{path}/{obj}')
data_files = []
for file in all_files:
    if file[-4:] == 'fits':
        data_files.append(file)
date = im.get_date(data_files[0])
# print(data_files[0])

freq = 1.
insert_value([obj, date, freq, data_files[0]])
