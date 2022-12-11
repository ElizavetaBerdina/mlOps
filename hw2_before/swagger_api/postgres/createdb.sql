CREATE TABLE public.data_regression (
  "Column_A"           NUMERIC(18,4),
  "Column_B"           NUMERIC(18,4),
  "Column_C"           NUMERIC(18,4),
  "Column_D"           NUMERIC(18,4),
  "Column_E"           NUMERIC(18,4),
  "target"             NUMERIC(18,4)
);

COPY public.data_regression(
  "Column_A",
  "Column_B",
  "Column_C",
  "Column_D",
  "Column_E",
  "target"
) FROM '/var/lib/postgresql/data/data_regression.csv' DELIMITER ';' CSV HEADER;

CREATE TABLE public.data_classification (
  "Column_A"           NUMERIC(18,4),
  "Column_B"           NUMERIC(18,4),
  "Column_C"           NUMERIC(18,4),
  "Column_D"           NUMERIC(18,4),
  "Column_E"           NUMERIC(18,4),
  "target"             INT()
);

COPY public.data_regression(
  "Column_A",
  "Column_B",
  "Column_C",
  "Column_D",
  "Column_E",
  "target"
) FROM '/var/lib/postgresql/data/data_classification.csv' DELIMITER ';' CSV HEADER;

CREATE TABLE public.models (
  "model_name"       TEXT PRIMARY KEY,
  "model_class"      TEXT NOT NULL,
  "params"           TEXT
);