# Pilot v2 solver data

These tables are generated from `ChinaUS_Gateway_Model_Ready_v2.xlsx` by
`scripts/build_pilot_from_workbook.py`.

## Unit conversion

The source workbook uses TEU. The solver uses FEU throughout:

- quantity and daily capacity: TEU / 2;
- arc, transfer and tardiness cost rates: USD/TEU × 2;
- service capacity per departure: daily FEU capacity × headway hours / 24.

The explicit eight-week departure calendar is stored in each service's
`departures_h` field and repeats after 1,344 hours. Extra source columns in the
CSV files preserve workbook sheet and row provenance; the model loader ignores
those columns.

`shipments.csv` contains three separate 40-batch cases. Use `--case D1`, `D2`
or `D3` to run Chicago, Memphis or Columbus independently.

`conversion_report.json` records the source workbook SHA-256, row counts,
conversion rules and integrity checks.
