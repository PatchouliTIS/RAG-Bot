#!/bin/bash
# Install postgres
apt install -y wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
apt update -y && apt install -y postgresql postgresql-contrib
# Install pgvector
apt install -y postgresql-server-dev-16
pushd /tmp && git clone --branch v0.6.1 https://github.com/pgvector/pgvector.git && pushd pgvector && make && make install && popd && popd
# Activate pgvector and the database
echo 'ray ALL=(ALL:ALL) NOPASSWD:ALL' | tee /etc/sudoers
service postgresql restart
# pragma: allowlist nextline secret
psql -U postgres -c "ALTER USER postgres with password 'postgres';"
psql -U postgres -c "CREATE EXTENSION vector;"
