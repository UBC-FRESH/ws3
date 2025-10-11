# Examples data layout

This folder contains small example datasets and run-generated artifacts used by the notebooks in `examples/`.

- `woodstock_model_files_tsa24_clipped/`, `woodstock_model_files_tsa22/`, `woodstock_model_files_tsa24/` – Woodstock-format text inputs for building a `ForestModel`.
- `libcbm_model_files/` – Minimal libCBM Standard Input Table scaffolding referenced by examples.
- `shp/tsa24_clipped.shp/stands.shp` – Stand polygons for the TSA24 toy dataset.
- `raster_spatial_demo/` – Generated at run time by the spatial allocation example (024). The notebook rasterizes the polygon stands into an inventory raster `tsa24_inventory.tif` here.
- `spatial_outputs/tsa24_demo/` – Generated at run time by the spatial allocation example (024). Per-year harvested rasters `harvested_YYYY.tif` are written here.

Notes
- These `raster_spatial_demo/` and `spatial_outputs/tsa24_demo/` directories are empty in the repository and populated when you run the spatial example.
- Empty placeholder `.gitkeep` files are used so that Git tracks the directories while ignoring generated files.
