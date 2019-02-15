host=gisdata-db.nina.no
database=gisdata

user=richard.hedger
schema=culvertfragmentation

psql -q -h $host -d $database -c "DROP SCHEMA \"${schema}\" CASCADE";

# CREATE USER "${user}" WITH
# 	LOGIN
# 	NOSUPERUSER
# 	NOCREATEDB
# 	NOCREATEROLE
# 	INHERIT
# 	NOREPLICATION
# 	CONNECTION LIMIT -1
# 	PASSWORD 'xxxxxx';

# GRANT gisuser TO \"${user}\";"

# Create schema in database and GRANT all rights to user
psql -q -h $host -d $database -c "CREATE SCHEMA \"${schema}\" AUTHORIZATION \"${user}\"";
psql -q -h $host -d $database -c "GRANT ALL ON SCHEMA \"${schema}\" TO \"${user}\"";

# Make schema readable for "gisuser" group (equally to everybody)
psql -q -h $host -d $database -c "GRANT USAGE ON SCHEMA \"${schema}\" TO gisuser";
psql -q -h $host -d $database -c "GRANT SELECT ON ALL TABLES IN SCHEMA \"${schema}\" TO gisuser"
psql -q -h $host -d $database -c "ALTER DEFAULT PRIVILEGES IN SCHEMA \"${schema}\" GRANT SELECT ON TABLES TO gisuser"


psql -h $host -d $database -c "DROP TABLE IF EXISTS \"${schema}\".\"CollatedData_v1_2019_02_12\";
CREATE TABLE \"${schema}\".\"CollatedData_v1_2019_02_12\" (
gid serial PRIMARY KEY,
source text,
year smallint,
vassdrag text,
station text,
site text,
\"coord_E_32N\" double precision,
\"coord_N_32N\" double precision,
oerret_aarsyngel_N double precision,
oerret_etteldre_N double precision,
geom geometry(Point, 25833));"

iconv -t UTF8 /data/R/Prosjekter/13704100_kartlegging_vandringshinder_for_sjoorret_gis_met/Bergan\ data_litteratur/CollatedData_v1_2019_02_12.csv | tail -n +2 | awk -v FS='\t' -v OFS='\t' '{gsub(",", ".", $8);gsub(",", ".", $9);print $1, $2, $3, $4, $5, $6, $7, $8, $9}' | psql -h $host -d $database -c "COPY \"${schema}\".\"CollatedData_v1_2019_02_12\" (
source,
year,
vassdrag,
station,
site,
\"coord_E_32N\",
\"coord_N_32N\",
oerret_aarsyngel_N,
oerret_etteldre_N) FROM STDIN;"

psql -h $host -d $database -c "UPDATE \"${schema}\".\"CollatedData_v1_2019_02_12\" SET geom = ST_Transform(ST_SetSRID(ST_Point(\"coord_E_32N\", \"coord_N_32N\"), 25832), 25833);"

psql -h $host -d $database -c "CREATE INDEX \"${schema}_CollatedData_v1_2019_02_12_spidx\" ON \"${schema}\".\"CollatedData_v1_2019_02_12\" USING gist (geom);"


\"Hydrography\".\"Norway_NVE_NedborfeltTilHav\"
psql -h $host -d $database -c "DROP VIEW IF EXISTS \"${schema}\".\"RegineWatersheds\" CASCADE;
CREATE OR REPLACE VIEW \"${schema}\".\"RegineWatersheds\" AS SELECT DISTINCT ON (a.gid, a.geom, a.vassdragsnummer, a.\"elvNavnHierarki\", a.\"nedborfeltTilHav\", a.\"overordnetNedborfeltNavn\", a.\"nedbfeltHavVassdragNr\", \"nedborfeltVassdragNrOverordnet\")
a.gid, a.geom, a.vassdragsnummer, a.\"elvNavnHierarki\", a.\"nedborfeltTilHav\", a.\"overordnetNedborfeltNavn\", a.\"nedbfeltHavVassdragNr\", \"nedborfeltVassdragNrOverordnet\"
FROM \"Hydrography\".\"Norway_NVE_RegineEnhet\" AS a
WHERE (CAST(split_part(a.vassdragsnummer, '.', 1) AS integer) IN (
SELECT DISTINCT ON (CAST(split_part(x.vassdragsnummer, '.', 1) AS integer)) CAST(split_part(x.vassdragsnummer, '.', 1) AS integer) FROM \"Hydrography\".\"Norway_NVE_RegineEnhet\" AS x,
\"${schema}\".\"CollatedData_v1_2019_02_12\" AS y
WHERE ST_DWithin(x.geom, y.geom, 0) -- AND \"nedborfeltTilHav\" != 'KYSTFELT'
)
-- OR ST_DWithin((SELECT geom FROM \"${schema}\".\"CollatedData_v1_2019_02_12\"), geom, 0)
)
-- The following watershed seems irrelevant though by ID/name) associated with NIDELVA
AND a.gid != 6882
;"

psql -h $host -d $database -c "DROP VIEW IF EXISTS \"${schema}\".\"Elvenett\";
CREATE OR REPLACE VIEW \"${schema}\".\"Elvenett\" AS SELECT DISTINCT ON (a.gid) a.* FROM
\"Hydrography\".\"Norway_NVE_Elvenett\" AS a,
\"${schema}\".\"RegineWatersheds\" AS b
WHERE ST_Intersects(a.geom, b.geom);"


bbox=$(psql -At -F' ' -h $host -d $database -c "SELECT ST_XMin(geom), ST_YMin(geom), ST_XMax(geom), ST_YMax(geom) FROM (SELECT ST_Buffer(ST_Collect(geom), 500) AS geom FROM \"${schema}\".\"RegineWatersheds\") AS x;")

bash ~/data_maintenance/Elevation_lidar2grass.sh dtm "$bbox" CulvertFragmentation_Elevation &> lidar2grass_dtm.log
