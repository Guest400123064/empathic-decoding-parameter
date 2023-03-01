import argparse

import pandas as pd
import sqlalchemy


def get_parser():

    parser = argparse.ArgumentParser(description='Export table from database')
    parser.add_argument("--database", "-d", type=str, required=True,
                        help='Database name')
    parser.add_argument("--table", "-t", type=str, required=True,
                        help='Table name')
    parser.add_argument("--output", "-o", type=str, required=True,
                        help='Output file name')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    db = sqlalchemy.engine.url.URL(drivername="mysql",
                                   host="localhost",
                                   database=args.database,
                                   query={"read_default_file": "~/.my.cnf", 
                                          "charset": "utf8"})
    engine = sqlalchemy.create_engine(db)

    df = pd.read_sql_table(args.table, engine)
    df.to_csv(args.output, index=False)
