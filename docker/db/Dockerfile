FROM postgres:18.1

# Create initialization script that will run on first container startup
# when the data directory is empty.
COPY scripts/init-db.sql /docker-entrypoint-initdb.d/

EXPOSE 5432
