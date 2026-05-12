-- init-db.sql

\getenv postgres_password POSTGRES_PASSWORD
\if :{?postgres_password}
\else
\echo POSTGRES_PASSWORD environment variable is required
\quit 1
\endif

-- 1) Ensure role 'noboom' exists (\gexec keeps password interpolation in psql)
SELECT format('CREATE ROLE noboom LOGIN PASSWORD %L', :'postgres_password')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'noboom'
);
\gexec

GRANT pg_read_all_settings TO noboom;

ALTER ROLE noboom WITH PASSWORD :'postgres_password';

-- 2) Create mlflow_db if it does not exist (psql-only trick with \gexec)
SELECT 'CREATE DATABASE mlflow_db OWNER noboom'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'mlflow_db'
);
\gexec

-- 3) Create optuna_db if it does not exist
SELECT 'CREATE DATABASE optuna_db OWNER noboom'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'optuna_db'
);
\gexec
